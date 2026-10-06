# Source this to run JAX on worker 0's four chips alone (tests, small benchmarks), without the
# other hosts: by default a process on a multi-host slice waits for all hosts.
export JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476 CLOUD_TPU_TASK_ID=0
export TPU_PROCESS_ADDRESSES=localhost:8476
