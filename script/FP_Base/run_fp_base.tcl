#!/usr/bin/env tclsh
# FP_Base — PyTorch backend, fp32 precision (no Triton kernel).
#
# Usage:
#   ./script/FP_Base/run_fp_base.tcl
#   ./script/FP_Base/run_fp_base.tcl task=bench
#   ./script/FP_Base/run_fp_base.tcl task=validate checkpoint=/path/to/model.pth.tar
#   ./script/FP_Base/run_fp_base.tcl task=train model=elsa_base_window8_256
#   ./script/FP_Base/run_fp_base.tcl family=sic_int
#   ./script/FP_Base/run_fp_base.tcl dry_run=1

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir ../..]]
set runner_sh  [file join $script_dir run_fp_base.sh]

# ---- 預設值 (可被 key=value 覆寫) ---------------------------
array set cfg {
    task                bench
    family              elsa_swin
    model               ""
    checkpoint          ""
    initial_checkpoint  ""
    device              cuda:0
    img_size            256
    batch               1
    val_batch           64
    train_batch         64
    warmup              5
    trials              20
    workers             4
    epochs              300
    split               val
    dry_run             0
}

# ---- 解析 key=value ------------------------------------------
foreach arg $argv {
    if {![regexp {^([^=]+)=(.*)$} $arg -> key value]} {
        puts stderr "Invalid argument '$arg'. Use key=value."
        exit 2
    }
    if {![info exists cfg($key)]} {
        puts stderr "Unknown key '$key'."
        exit 2
    }
    set cfg($key) $value
}

# ---- 印出摘要 -----------------------------------------------
puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]  \[FP_Base\]"
puts "backend=pytorch  dtype=fp32  task=$cfg(task)  family=$cfg(family)"

# ---- 組合給 run_fp_base.sh 的環境變數 -----------------------
set env_pairs [list \
    RUN_TASK=$cfg(task) \
    RUN_FAMILY=$cfg(family) \
    RUN_MODEL=$cfg(model) \
    RUN_CHECKPOINT=$cfg(checkpoint) \
    RUN_INITIAL_CHECKPOINT=$cfg(initial_checkpoint) \
    RUN_DEVICE=$cfg(device) \
    RUN_IMG_SIZE=$cfg(img_size) \
    RUN_BATCH=$cfg(batch) \
    RUN_VAL_BATCH=$cfg(val_batch) \
    RUN_TRAIN_BATCH=$cfg(train_batch) \
    RUN_WARMUP=$cfg(warmup) \
    RUN_TRIALS=$cfg(trials) \
    RUN_WORKERS=$cfg(workers) \
    RUN_EPOCHS=$cfg(epochs) \
    RUN_SPLIT=$cfg(split) \
    RUN_DRY_RUN=$cfg(dry_run)]

set cmd [concat [list env] $env_pairs [list bash $runner_sh]]
puts "command: [join $cmd " "]"

if {$cfg(dry_run)} { exit 0 }

if {[catch {exec {*}$cmd >@ stdout 2>@ stderr} result opts]} {
    set ec [dict get $opts -errorcode]
    if {[lindex $ec 0] eq "CHILDSTATUS"} {
        exit [lindex $ec 2]
    }
    puts stderr $result
    exit 1
}
