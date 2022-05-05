#!/usr/bin/env python
# coding=utf-8

import logging
import time
import typing
import datetime

import GPUtil


class Job:
    def __init__(self,
                 name,
                 priority=0,
                 device=0,
                 required_gpu=0,
                 creation_ts=datetime.datetime.now()):
        self.name_ = name
        # priority 0 high, 1 low
        self.priority_ = priority
        self.device_ = device  # 目前仅支持单卡
        self.required_gpu_ = required_gpu  # 单位 Mb
        self.allocated_gpu_ = 0  # 单位 Mb
        self.creation_ts_ = creation_ts  # 创建时间戳


class NvidiaDeviceMonitor:
    # TODO 看到的是物理卡信息，作业用的虚拟卡的话，需要有资源折算
    def __init__(self):
        pass

    def get_device_view(self) -> list:
        # ref: https://github.com/anderskm/gputil
        return GPUtil.getGPUs()

    def show_all_status(self, version, jobs=[]):
        gpus = self.get_device_view()
        g_count = len(gpus)
        p_count = [0] * g_count
        for j in jobs:
            p_count[j.device_] += 1
        for idx in range(g_count):
            g = gpus[idx]
            logging.info(
                'version: {}, gpu_id: {}, process-count: {}, mem-util: {}%, sm-util: {}%'
                .format(version, g.id, p_count[idx],
                        format(g.memoryUtil * 100, '.4f'),
                        format(g.load * 100, '.4f')))


class ResourceManager:
    def __init__(self):
        self.jobs_ = dict()  # dict, job_name -> job_info
        self.gpu_topo_ = []
        self.gpu_manager_ = NvidiaDeviceMonitor()  # 暂不使用
        self.version_ = self.__get_timestamp_ms()  # version

    def __get_timestamp_ms(self):
        return int(time.time() * 1000)

    def update_jobs(self, jobs=[], gpus=[]) -> int:
        '''
        更新作业列表
        '''
        new_jobs = dict()
        changed = False
        for j in jobs:
            new_jobs[j.name_] = j
            if j.name_ not in self.jobs_.keys():
                changed = True
        for j in self.jobs_.keys():
            if j not in new_jobs.keys():
                changed = True
            elif self.jobs_[j].device_ != new_jobs[j].device_:
                changed = True
            elif self.jobs_[j].required_gpu_ != new_jobs[j].required_gpu_:
                changed = True
        if changed == False:
            return self.version_
        self.gpu_topo_ = gpus
        self.jobs_ = new_jobs
        self.version_ = self.__get_timestamp_ms()
        return self.version_

    def get_view(self) -> dict:
        '''
        获取显卡分配视图
        '''
        KEY = 'gpuConfigInfo'
        view = dict({KEY: {}, 'version': self.version_})
        view[KEY] = self.__resource_decide()
        logging.info('new gpu resource distribution: {}'.format(view))
        return view

    def __distribute_one_gpu(self,
                             gpu_info,
                             jobs=[]) -> typing.Tuple[dict, bool]:
        '''
        简化版单卡资源分配策略：
        1. 目前仅有两优先级, 高优0, 低优1
        2. 高优作业按照 required 分配, 上层调度可能存在超发
        3. 低优作业在剩余资源池中按 required 比例分配
        4. 循环上述过程
        '''
        def priority(job) -> int:
            return job.priority_

        def creationTimestamp(job) -> datetime.datetime:
            return job.creation_ts_

        jobs.sort(key=priority)
        gpu_memory_total = gpu_info['Memory']
        high_jobs = list(filter(lambda j: j.priority_ == 0, jobs))
        high_jobs.sort(key=creationTimestamp)
        low_jobs = list(filter(lambda j: j.priority_ != 0, jobs))

        logging.info(
            'there are {} jobs running on GPU {}, high {}, low {}.'.format(
                len(jobs), gpu_info['Index'], len(high_jobs), len(low_jobs)))

        distribution = dict()

        # 高优作业的 required 不能超过总和
        high_required = sum([j.required_gpu_ for j in high_jobs])
        low_required = sum([j.required_gpu_ for j in low_jobs])
        over_max = high_required > gpu_memory_total
        if over_max == True:
            logging.warn(
                'not enough gpu memory for high priority jobs, total {}, required {}'
                .format(gpu_memory_total, high_required))
            # return distribution, False

        left_mem = gpu_memory_total
        for j in high_jobs:
            distribution[j.name_] = {
                'priority': j.priority_,
                'device_id': 0,  # 单卡是容器内编号都是 0 开始
                'host_device_id': gpu_info['Index'],
                'maxDeviceMemMb': 1  # default value, 0 means no limit...
            }
            if over_max == False:
                # 未超发
                if left_mem >= j.required_gpu_:
                    distribution[j.name_]['maxDeviceMemMb'] = j.required_gpu_
                    left_mem = left_mem - j.required_gpu_
            else:
                # 已超发
                left_mem = 0
                allocated = float(
                    j.required_gpu_) / high_required * gpu_memory_total
                allocated = int(allocated)
                distribution[j.name_]['maxDeviceMemMb'] = min(
                    allocated, j.required_gpu_)

        logging.info('left {}M gpu mem for low priority {} jobs'.format(
            left_mem, len(low_jobs)))
        for j in low_jobs:
            allocated = 1  # default value 1M, maybe not good
            if left_mem != 0:
                if low_required == 0:
                    allocated = 1.0 / len(low_jobs) * left_mem
                else:
                    allocated = float(j.required_gpu_) / low_required * left_mem
            allocated = int(allocated)
            if allocated == 0:
                allocated = 1
            distribution[j.name_] = {
                'priority': j.priority_,
                'device_id': 0,
                'host_device_id': gpu_info['Index'],
                'maxDeviceMemMb': min(allocated, j.required_gpu_)
            }

        return distribution, True

    def __resource_decide(self) -> dict():
        '''
        所有卡上所有作业的资源分配决策
        '''
        # gpus = self.gpu_manager_.get_device_view()
        gnum = len(self.gpu_topo_)
        # gpu id 为从 0 开始的连续编号
        # topo = [[job1, job2], [job3, job4] ...]
        # topo[idx] 表示跑在 idx 号 gpu 上的作业列表
        topo = [[] for i in range(gnum)]
        for jk in self.jobs_:
            j = self.jobs_[jk]
            gid = j.device_
            if gid < 0 or gid >= gnum:
                logging.warn('job {} has invalid gpuid {}, gpu num {}'.format(
                    j.name_, gid, gnum))
                continue
            topo[gid].append(j)

        dist_all = dict()
        for gid in range(gnum):
            if len(topo[gid]) == 0:
                logging.info('no jobs on gpu {}, skipped'.format(gid))
            dist, ok = self.__distribute_one_gpu(self.gpu_topo_[gid], topo[gid])
            if ok == False:
                continue
            # merge, Python 3.5+ required
            dist_all = {**dist_all, **dist}
        return dist_all
