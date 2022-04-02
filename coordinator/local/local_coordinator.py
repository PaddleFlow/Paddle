#!/usr/bin/env python
# coding=utf-8

import socket
import string
import threading
import time
import logging
import json
import argparse
import configparser
import sys
import signal

from kubernetes import client as k8s_cli
from kubernetes import config as k8s_cfg

from resource_manager import ResourceManager, Job


class LocalCoordinator(threading.Thread):
    def __init__(self, config=None):
        if config == None:
            logging.error('no config')
            return
        section = config['local_coordinator']

        threading.Thread.__init__(self)
        self.hostname_ = section['hostname']
        self.resource_manager_ = ResourceManager()
        self.gpu_mem_mb_per_cgpu_share_ = 100  # default 100M per cgpu share
        self.gpu_conf_file = section['gpu_config_file']
        self.job_viwe_version_ = 0
        self.run_flag_ = True
        self.mutex_ = threading.Lock()

        k8s_cfg.load_kube_config()
        self.k8s_api_ = k8s_cli.CoreV1Api()

    def __del__(self):
        logging.info('clean up now')
        del self.resource_manager_

    def __get_self_node(self) -> string:
        return socket.gethostname()

    def __filter_job(self, pod) -> bool:
        if pod.spec.node_name != self.hostname_:
            return False
        VALID_STATUS = ['ContainerCreating', 'Running']
        if pod.status.phase not in VALID_STATUS:
            return False
        annotations = pod.metadata.annotations
        if annotations == None:
            return False
        if 'BAIDU_COM_GPU_IDX' not in annotations.keys():
            return False
        if 'antman/job-name' not in annotations.keys():
            return False
        containers = pod.spec.containers
        if len(containers) != 1:
            logging.warning(
                'pod {} has {} containers, only support 1 container now'.format(
                    pod.metadata.name, len(containers)))
            return False
        return True

    def __parse_job(self, pod) -> Job:
        PRIO_KEY = 'antman/priority'
        annotations = pod.metadata.annotations
        name = annotations['antman/job-name']
        priority = 1  # default low priority
        if PRIO_KEY in annotations.keys():
            priority = annotations[PRIO_KEY]
        device_id = int(annotations['BAIDU_COM_GPU_IDX'])
        container = pod.spec.containers[0]
        required_gpu_mem = 100  # default 100M
        if container.resources != None and container.resources.requests != None:
            if 'baidu.com/cgpu_memory' in container.resources.requests.keys():
                requests = container.resources.requests
                required_gpu_mem = int(requests['baidu.com/cgpu_memory']
                                       ) * self.gpu_mem_mb_per_cgpu_share_
        logging.info(
            'find job {}, priority {}, device_id {}, required_gpu_men {}M'.
            format(name, priority, device_id, required_gpu_mem))
        return Job(name, priority, device_id, required_gpu_mem)

    def __get_gpu_topo(self) -> list:
        GPU_KEY = 'kubernetes.io/baidu-cgpu.gpu-topo'
        logging.info('Get node {} info'.format(self.hostname_))
        ret = self.k8s_api_.read_node(self.hostname_)
        if GPU_KEY not in ret.metadata.annotations.keys():
            return []

        def index(gpu):
            # print(gpu)
            return gpu['Index']

        gpu_topo = json.loads(ret.metadata.annotations[GPU_KEY])
        gpu_topo.sort(key=index)
        return gpu_topo

    def __get_jobs(self) -> list:
        # Configs can be set in Configuration class directly or using helper utility
        logging.info('Listing pods ...')
        ret = self.k8s_api_.list_pod_for_all_namespaces(watch=False)

        jobs = []
        for i in ret.items:
            if self.__filter_job(i) == False:
                continue
            j = self.__parse_job(i)
            jobs.append(j)
        return jobs

    def __dump_decision(self, view):
        content = json.dumps(view, indent=4, sort_keys=True)
        logging.info('dump decision: {}'.format(content))
        with open(self.gpu_conf_file, 'w') as f:
            f.write(content)
            f.truncate()
            f.close()

    def stop(self):
        logging.info('ready to exit now')
        self.run_flag_ = False

    def run(self):
        while (self.run_flag_):
            time.sleep(3)
            # 1. get gpu info
            gpus = self.__get_gpu_topo()
            # 2. watch pods
            jobs = self.__get_jobs()
            # 3. 更新作业情况
            ver = self.resource_manager_.update_jobs(jobs, gpus)
            if ver == self.job_viwe_version_:
                # unchanged
                logging.info('unchanged, ignore')
                continue
            self.job_viwe_version_ = ver
            # 4. 获取决策信息
            view = self.resource_manager_.get_view()
            # 5. dump 决策
            self.__dump_decision(view)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-c',
                      '--config_file',
                      help='config file for local coordinator',
                      default='config/config.conf')
    args.add_argument('-f',
                      '--log_file',
                      help='log file',
                      default='log/coordinator.log')

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('load config {}, logfile {}'.format(args.config_file, args.log_file))
    # TODO 日志切割清理
    logging.basicConfig(filename=args.log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print('running...')
    confpar = configparser.ConfigParser()
    confpar.read(args.config_file)
    logging.info('local coordinator running...')

    global co
    co = LocalCoordinator(confpar)

    def signal_handler(sig, frame):
        print('\nPressed Ctrl+C!')
        global co
        co.stop()
        co.join()
        del co

    signal.signal(signal.SIGINT, signal_handler)

    co.start()
