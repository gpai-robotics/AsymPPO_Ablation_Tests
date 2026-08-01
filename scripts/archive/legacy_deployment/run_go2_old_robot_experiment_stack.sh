#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_IFACE="${GO2_DDS_IFACE:-enp0s31f6}"
ROLE="${1:-help}"
NET_IFACE="${2:-${DEFAULT_IFACE}}"
EXPERIMENT_PARAMS_DIR="${ROOT_DIR}/reference_repos/unitree_rl_lab_go2_old_robot_experiments/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params"

usage() {
  cat <<EOF
Usage:
  scripts/deploy/run_go2_old_robot_experiment_stack.sh odom [iface]
  scripts/deploy/run_go2_old_robot_experiment_stack.sh bridge [iface]
  scripts/deploy/run_go2_old_robot_experiment_stack.sh ctrl [iface]
  scripts/deploy/run_go2_old_robot_experiment_stack.sh summary [latest|logfile]
  scripts/deploy/run_go2_old_robot_experiment_stack.sh help

Roles:
  odom     Laucd /home/bhuvan/projects/rma/rma_go2_lab
scripts/deploy/run_go2_old_robot_experiment_stack.sh summary latest
{
  "counts": {
    "fsm_transitions": 3,
    "joint_err": 196,
    "joint_raw_action": 196,
    "joint_rel_cmd": 196,
    "joint_rel_pos": 196,
    "obs_samples": 196,
    "velocity_cmd": 196
  },
  "fsm": {
    "entered_velocity": true,
    "returned_to_passive": true,
    "transitions": [
      {
        "from": "Passive",
        "to": "FixStand"
      },
      {
        "from": "FixStand",
        "to": "Velocity"
      },
      {
        "from": "Velocity",
        "to": "Passive"
      }
    ]
  },
  "joint_tracking": {
    "joint_err_leg_axis_mean_abs": {
      "FL": [
        0.07893367346938776,
        0.11814795918367348,
        0.24581122448979592
      ],
      "FR": [
        0.052479591836734694,
        0.1204030612244898,
        0.26313775510204085
      ],
      "RL": [
        0.0765,
        0.06383673469387756,
        0.19183163265306122
      ],
      "RR": [
        0.0880969387755102,
        0.060979591836734695,
        0.18053571428571427
      ]
    },
    "joint_err_leg_norm_max": {
      "FL": 0.673887230922207,
      "FR": 0.7412725544629317,
      "RL": 0.5756161915721273,
      "RR": 0.4438670972261855
    },
    "joint_err_leg_norm_mean": {
      "FL": 0.3044162449454514,
      "FR": 0.31556537337028484,
      "RL": 0.2348873147899279,
      "RR": 0.22292567494136
    },
    "mean_left_minus_right": {
      "x": 0.007448979591836735,
      "y": 0.00034183673469387777,
      "z": -0.003025510204081633
    },
    "raw_action_leg_norm_mean": {
      "FL": 1.1552007292744484,
      "FR": 1.171128205436634,
      "RL": 0.9566625264584314,
      "RR": 1.195354687231452
    },
    "rel_cmd_leg_norm_mean": {
      "FL": 0.28137371895237995,
      "FR": 0.2812927016172556,
      "RL": 0.2363774097305804,
      "RR": 0.2957588143532973
    },
    "rel_pos_leg_norm_mean": {
      "FL": 0.12626502261748906,
      "FR": 0.16052487298291662,
      "RL": 0.22509301297938386,
      "RR": 0.1487541103525372
    },
    "side_abs_err_max": {
      "left": [
        0.345,
        0.277,
        0.422
      ],
      "right": [
        0.24,
        0.283,
        0.49
      ]
    },
    "side_abs_err_mean": {
      "left": [
        0.07774489795918367,
        0.09101530612244897,
        0.2188061224489796
      ],
      "right": [
        0.07029591836734694,
        0.0906734693877551,
        0.22183163265306122
      ]
    }
  },
  "log_file": "/home/bhuvan/projects/rma/rma_go2_lab/logs/go2_ctrl_expe
riment/go2_ctrl_experiment_20260529_145509.log",                         "obs": {
    "base_ang_norm": {
      "count": 196,
      "max_abs": 1.412012039608728,
      "mean_abs": 0.3475639570680606,
      "p95_abs": 0.8485185907682028
    },
    "cmd_windows": {
      "vx": {
        "near_zero": {
          "count": 135,
          "max_abs": 0.047,
          "mean_abs": 0.0033259259259259258,
          "p95_abs": 0.038
        },
        "negative": {
          "count": 29,
          "max_abs": 0.15,
          "mean_abs": 0.12386206896551724,
          "p95_abs": 0.15
        },
        "positive": {
          "count": 32,
          "max_abs": 0.2,
          "mean_abs": 0.124125,
          "p95_abs": 0.2
        }
      },
      "vy": {
        "near_zero": {
          "count": 164,
          "max_abs": 0.049,
          "mean_abs": 0.0016402439024390245,
          "p95_abs": 0.0
        },
        "negative": {
          "count": 17,
          "max_abs": 0.2,
          "mean_abs": 0.12629411764705883,
          "p95_abs": 0.2
        },
        "positive": {
          "count": 15,
          "max_abs": 0.2,
          "mean_abs": 0.1270666666666667,
          "p95_abs": 0.1986
        }
      },
      "wz": {
        "near_zero": {
          "count": 108,
          "max_abs": 0.049,
          "mean_abs": 0.0023425925925925927,
          "p95_abs": 0.02144999999999972
        },
        "negative": {
          "count": 47,
          "max_abs": 0.907,
          "mean_abs": 0.42459574468085104,
          "p95_abs": 0.8628999999999999
        },
        "positive": {
          "count": 41,
          "max_abs": 0.848,
          "mean_abs": 0.4242926829268293,
          "p95_abs": 0.818
        }
      }
    },
    "gravity_xy_tilt": {
      "count": 196,
      "max_abs": 0.14052757736472937,
      "mean_abs": 0.06724130351537838,
      "p95_abs": 0.12047261544575459
    }
  },
  "operator_flags": {
    "use_intervention_notes": true,
    "use_surface_notes": true,
    "use_video_review": true
  },
  "velocity": {
    "filtered_max_abs": {
      "vx": 0.2,
      "vy": 0.2,
      "wz": 0.907
    },
    "filtered_windows": {
      "vx": {
        "near_zero": {
          "count": 135,
          "max_abs": 0.047,
          "mean_abs": 0.0033259259259259258,
          "p95_abs": 0.038
        },
        "negative": {
          "count": 29,
          "max_abs": 0.15,
          "mean_abs": 0.12386206896551724,
          "p95_abs": 0.15
        },
        "positive": {
          "count": 32,
          "max_abs": 0.2,
          "mean_abs": 0.1226875,
          "p95_abs": 0.2
        }
      },
      "vy": {
        "near_zero": {
          "count": 165,
          "max_abs": 0.049,
          "mean_abs": 0.0019212121212121213,
          "p95_abs": 0.0
        },
        "negative": {
          "count": 16,
          "max_abs": 0.2,
          "mean_abs": 0.1285,
          "p95_abs": 0.2
        },
        "positive": {
          "count": 15,
          "max_abs": 0.2,
          "mean_abs": 0.1270666666666667,
          "p95_abs": 0.1986
        }
      },
      "wz": {
        "near_zero": {
          "count": 108,
          "max_abs": 0.049,
          "mean_abs": 0.0023425925925925927,
          "p95_abs": 0.02144999999999972
        },
        "negative": {
          "count": 47,
          "max_abs": 0.907,
          "mean_abs": 0.42459574468085104,
          "p95_abs": 0.8628999999999999
        },
        "positive": {
          "count": 41,
          "max_abs": 0.848,
          "mean_abs": 0.423,
          "p95_abs": 0.818
        }
      }
    },
    "lin_vel_max_abs": {
      "imu_wz": 0.855,
      "vx": 0.28,
      "vy": 0.165
    },
    "zero_command_drift": {
      "filtered_cmd_reference": {
        "filtered_vx_near_zero": {
          "count": 135,
          "max_abs": 0.047,
          "mean_abs": 0.0033259259259259258,
          "p95_abs": 0.038
        },
        "filtered_vy_near_zero": {
          "count": 165,
          "max_abs": 0.049,
          "mean_abs": 0.0019212121212121213,
          "p95_abs": 0.0
        },
        "filtered_wz_near_zero": {
          "count": 108,
          "max_abs": 0.049,
          "mean_abs": 0.0023425925925925927,
          "p95_abs": 0.02144999999999972
        }
      },
      "measured_when_all_cmd_axes_near_zero": {
        "imu_wz": {
          "count": 33,
          "max_abs": 0.263,
          "mean_abs": 0.06781818181818182,
          "p95_abs": 0.2275999999999999
        },
        "lin_vel_vx": {
          "count": 33,
          "max_abs": 0.28,
          "mean_abs": 0.06363636363636364,
          "p95_abs": 0.18279999999999996
        },
        "lin_vel_vy": {
          "count": 33,
          "max_abs": 0.154,
          "mean_abs": 0.027636363636363636,
          "p95_abs": 0.10559999999999996
        },
        "lin_vel_xy_norm": {
          "count": 33,
          "max_abs": 0.28,
          "mean_abs": 0.08074811977465735,
          "p95_abs": 0.19931723828843262
        }
      },
      "measured_when_vx_cmd_near_zero": {
        "lin_vel_vx": {
          "count": 135,
          "max_abs": 0.28,
          "mean_abs": 0.03921481481481482,
          "p95_abs": 0.16369999999999998
        }
      },
      "measured_when_vy_cmd_near_zero": {
        "lin_vel_vy": {
          "count": 165,
          "max_abs": 0.165,
          "mean_abs": 0.019278787878787878,
          "p95_abs": 0.11179999999999982
        }
      },
      "measured_when_wz_cmd_near_zero": {
        "imu_wz": {
          "count": 108,
          "max_abs": 0.35,
          "mean_abs": 0.055574074074074074,
          "p95_abs": 0.16489999999999994
        }
      }
    }
  }
}
nch the ROS2 odometry stack for the old robot.
  bridge   Bridge /odometry/filtered to udp://127.0.0.1:5560.
  ctrl     Force experimental deploy.yaml to odometry mode, then launch experimental go2_ctrl with logging.
  summary  Summarize the latest experimental go2_ctrl log or a specific logfile.
EOF
}

run_odom() {
  cd "${ROOT_DIR}"
  export GO2_DDS_IFACE="${NET_IFACE}"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/deploy/source_local_odom_env.sh"
  ros2 launch go2_odometry go2_odometry_switch.launch.py odom_type:=use_full_odom
}

run_bridge() {
  cd "${ROOT_DIR}"
  export GO2_DDS_IFACE="${NET_IFACE}"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/scripts/deploy/source_local_odom_env.sh"
  python "${ROOT_DIR}/scripts/deploy/bridge_odometry_to_udp.py"
}

run_ctrl() {
  cd "${ROOT_DIR}"
  python "${ROOT_DIR}/scripts/deploy/set_base_lin_vel_source.py" --params-dir "${EXPERIMENT_PARAMS_DIR}" --source odometry
  "${ROOT_DIR}/scripts/deploy/run_go2_ctrl_logged_experiment.sh" "${NET_IFACE}"
}

run_summary() {
  cd "${ROOT_DIR}"
  local target="${1:-latest}"
  if [ "${target}" = "latest" ]; then
    target="$(ls -t "${ROOT_DIR}"/logs/go2_ctrl_experiment/*.log 2>/dev/null | head -n 1 || true)"
    if [ -z "${target}" ]; then
      echo "[ERROR] No experimental go2_ctrl logs found under ${ROOT_DIR}/logs/go2_ctrl_experiment" >&2
      exit 1
    fi
  fi
  python "${ROOT_DIR}/scripts/deploy/summarize_go2_ctrl_log.py" --log-file "${target}"
}

case "${ROLE}" in
  odom)
    run_odom
    ;;
  bridge)
    run_bridge
    ;;
  ctrl)
    run_ctrl
    ;;
  summary)
    run_summary "${2:-latest}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown role: ${ROLE}" >&2
    usage >&2
    exit 1
    ;;
esac
