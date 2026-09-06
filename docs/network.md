# Network and FCI access

How this workstation reaches the FR3, and which address the FCI actually lives
on.

> Carried over from the `tekken_franka_ros2_ws` setup on this machine, where it
> was measured against **Franka-Tekken-00** (robot system 5.10.0). If you are
> reading this on a different workstation or a different arm, treat the
> addresses as an example and re-run the checks at the bottom.

## The short version

The Control unit is on the **company network at `10.7.7.7`**. That is where the
FCI is, and that is the address every command in this workspace should use. The
workstation is on the same company network by wire, so nothing else is needed:
no crossover cable, no static addressing, no second interface.

```bash
ros2 run inspire_franka_bringup fci_check 10.7.7.7
```

That check uses libfranka directly and no ROS at all, so it answers "is the FCI
reachable" even when the workspace does not build.

## Addresses

| | address | serves | use it for |
|---|---|---|---|
| Company network (wired) | `10.7.7.7` | Desk (443) **and FCI (1337)** | **everything** |
| Workstation | DHCP in `10.7.128.0/17` | — | — |
| X5 robot-net, RJ45 on the **arm base** | `192.168.69.1` | Desk only | nothing here |

Both robot addresses serve the Desk web UI, which is what makes this confusing.
Only the company-network address serves the FCI. Franka's manual is explicit:
*"FCI mode cannot be used when the robot is connected via the X5 – Robot network
on the robot foot."*

Two consequences worth knowing before copying a command from elsewhere:

- **`robot.franka.de` does not resolve on the company network.** That name is
  served by the robot's own DHCP/DNS on the X5 net. Use the literal address.
- **`192.168.69.1:1337` will never answer.** If you are debugging why, you are
  on the wrong network — it is not a licence or activation problem.

This is also why `arm.launch.py` has **no default** for `robot_ip`. Franka's own
default is `172.16.0.3`, which is the robot's private network and wrong here;
guessing wastes more time than asking.

## The router hop

The workstation and the robot are on the same company network but different
subnets, so traffic crosses one router hop — `ping` returns `ttl=63` rather than
`64`, and the robot never appears in `ip neigh`.

That is fine in practice; the link measured sub-millisecond and lossless, which
is what a 1 kHz FCI loop needs. But it does mean the path is not purely a
switch, so if `communication_constraints_violation` ever starts appearing, the
corporate link is a real suspect and not just a missing real-time kernel.
Re-measure before blaming controller configuration:

```bash
ping -c 20 10.7.7.7
```

## Checks

**Is the port open?** No ROS, no container:

```bash
for p in 443 1337; do
  timeout 4 bash -c "echo > /dev/tcp/10.7.7.7/$p" 2>/dev/null \
    && echo "10.7.7.7:$p OPEN" || echo "10.7.7.7:$p closed/unreachable"
done
```

`1337 OPEN` means the FCI is activated and reachable. Closed means either the
FCI is off in Desk, or you are not on the company network.

**Is the robot ready?** No login needed on this endpoint:

```bash
curl -sk https://10.7.7.7/admin/api/system-status | python3 -c "
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
