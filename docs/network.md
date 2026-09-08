# Network and FCI access

How this workstation reaches the FR3, and which address the FCI actually lives
on.

This setup was verified against **Franka-Tekken-00** (robot system 5.10.0).
The Control C2 port and workstation Ethernet interface are connected directly,
without the company network in the path.

## The short version

The Control C2 interface has the static address **`172.16.0.2/24`**. The
workstation Ethernet interface has **`172.16.0.1/24`**. Every FCI command in
this workspace should therefore use `172.16.0.2`.

```bash
ros2 run inspire_franka_bringup fci_check 172.16.0.2
```

That check uses libfranka directly and no ROS at all, so it answers "is the FCI
reachable" even when the workspace does not build.

## Addresses

| | address | serves | use it for |
|---|---|---|---|
| Control C2, directly cabled to workstation | `172.16.0.2/24` | Desk (443) **and FCI (1337)** | **FCI and Desk** |
| Workstation Ethernet (`Franka FCI Direct`) | `172.16.0.1/24` | — | direct robot link |
| X5 robot-net, RJ45 on the **arm base** | `192.168.0.1` | Desk only | initial setup/reconfiguration |

Both robot addresses serve the Desk web UI, which is what makes this confusing.
Only the Control C2 address serves the FCI. Franka's manual is explicit:
*"FCI mode cannot be used when the robot is connected via the X5 – Robot network
on the robot foot."*

Two consequences worth knowing before copying a command from elsewhere:

- **`robot.franka.de` belongs to the X5 setup network.** It is not the FCI
  address; use the literal Control address on the direct link.
- **The X5 address on port 1337 will never answer.** If you are debugging why, you are
  on the wrong network — it is not a licence or activation problem.

The launch files in this workspace default `robot_ip` to `172.16.0.2`.

## Workstation profile

The NetworkManager profile is named `Franka FCI Direct` and is bound to
`enp0s31f6`. It uses manual IPv4 address `172.16.0.1/24`, no gateway, no DNS,
no extra routes, IPv6 disabled, and `ipv4.never-default yes`. Activate that
profile after connecting the cable to Control C2:

```bash
nmcli connection up "Franka FCI Direct"
ip -brief -4 address show enp0s31f6
ip route get 172.16.0.2
ping -I enp0s31f6 -c 20 172.16.0.2
```

Expect `172.16.0.1/24` on the interface and a route resembling
`172.16.0.2 dev enp0s31f6 src 172.16.0.1`, with no `via` gateway. The static
Control address does not expire or change while this configuration remains
saved in Desk.

The direct cable removes the company router and DHCP allocation from the FCI
path. It does not replace the real-time kernel needed for reliable 1 kHz
control.

To return to X5 setup, move the cable to X5 and activate the ordinary DHCP
profile. Do not use the direct static profile on X5.

## Checks

First verify ordinary network reachability:

```bash
ping -I enp0s31f6 -c 20 172.16.0.2
```

**Is the port open?** No ROS, no container:

```bash
for p in 443 1337; do
  timeout 4 bash -c "echo > /dev/tcp/172.16.0.2/$p" 2>/dev/null \
    && echo "172.16.0.2:$p OPEN" || echo "172.16.0.2:$p closed/unreachable"
done
```

`1337 OPEN` means the FCI is activated and reachable. Closed means either the
FCI is off in Desk, the direct profile is inactive, or the cable is not on C2.

**Is the robot ready?** No login needed on this endpoint:

```bash
curl -sk https://172.16.0.2/admin/api/system-status | python3 -c "
import sys, json
d = json.load(sys.stdin); s = d['safety']
print('fci   :', d['controlToken']['fciActive'])
print('token :', d['controlToken']['activeToken'])
print('mode  :', d['derived']['operatingMode'], '| light:', d['derived']['desiredColor']['color'])
print('brakes:', set(s['brakeState']), '| sto:', s['stoState'])
print('errors:', d['robot']['robotErrors'])"
```

You want `fci: True`, `mode: Execution`, `light: Green`, `brakes: {'Unlocked'}`,
`errors: []`. `token.ownedBy` shows who currently holds control — one session at
a time, so if it names someone else, that is your answer.

Related endpoints on the same host: `/admin/api/features` lists licensed
features, `/admin/api/system-version` returns the system version.

## Containers

`docker/docker-compose.yml` uses `network_mode: host`, so the container shares
the workstation's interfaces and reaches the arm with no extra configuration.
There is no port mapping to get wrong and no bridge address to translate.

The same compose file adds `cap_add: [SYS_NICE]` and raises `rtprio`/`memlock`,
which is what the FCI's control thread needs. Note that the **hand** needs none
of this — it is a 50 Hz serial device, and shares the container only for
convenience.
