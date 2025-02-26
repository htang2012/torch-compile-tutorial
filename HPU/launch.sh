 PT_HPU_LAZY_MODE=0 python -Xfrozen_modules=off -m debugpy --listen localhost:5678 --wait-for-client main_torch_compile.py --device hpu --torch_compile_type hpubackend
