#!/usr/bin/env bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TERM="xterm-256color"

alias ll='ls -halF'
alias la='ls -A'
alias l='ls -CF'

if [ -x /usr/bin/dircolors ]; then
  eval "$(dircolors -b)"
  alias ls='ls --color=auto'
  alias grep='grep --color=auto'
fi

if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

export PS1="\[\033[01;32m\]\u@inspire_franka:\w\$\[\033[00m\] "

export ROS_DOMAIN_ID=42
export RCUTILS_COLORIZED_OUTPUT=1

source /opt/ros/"${ROS_DISTRO:-jazzy}"/setup.bash
if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then
  source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
fi

# Build the workspace and source it.
rg2() {
  if [[ -d /root/develop_ws/src ]]; then
    ( cd /root/develop_ws && colcon build --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release ) || return $?
  fi
  if [[ -f /root/develop_ws/install/setup.bash ]]; then
    source /root/develop_ws/install/setup.bash
  fi
}

# Run the workspace's own tests - the ones that need neither hardware nor a
# simulator. See the repo README.
rg2test() {
  ( cd /root/develop_ws \
    && python3 -m pytest -q apps/operations \
    && colcon test --packages-select \
         inspire_hand_driver inspire_hand_description \
         inspire_franka_description inspire_franka_sim \
         inspire_franka_trajectory_replay franka_trajectory_replay \
         camera_calibration \
    && colcon test-result --verbose )
}

# Source the overlay on shell start if it has already been built, so a fresh
# `docker exec` can run nodes without needing `rg2` first.
if [[ -f /root/develop_ws/install/setup.bash ]]; then
  source /root/develop_ws/install/setup.bash
fi
