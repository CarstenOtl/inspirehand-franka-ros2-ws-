#!/usr/bin/env python3
# Copyright (c) 2026 Agile Robots SE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the robot will accept FCI torque control, before anything is launched.

Read-only: it only queries the Desk status endpoint, which needs no authentication and takes no
control token. Worth running first, because franka_hardware does not catch libfranka's
ControlException - a robot that is not in Execution mode takes the whole ros2_control_node down
with a SIGABRT and a stalled launch rather than a readable error.
"""

import argparse
import json
import ssl
import sys
import urllib.request

STATUS_PATH = '/admin/api/system-status'


def fetch(host, timeout):
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    url = 'https://%s%s' % (host, STATUS_PATH)
    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode())


def check(status):
    """Return (name, ok, detail, hint) for every precondition FCI control needs."""
    safety = status.get('safety', {})
    derived = status.get('derived', {})
    token = status.get('controlToken', {})
    inputs = safety.get('safeInputState', {})

    brakes = safety.get('brakeState', [])
    mode = derived.get('operatingMode')
    controller_status = safety.get('safetyControllerStatus')
    guiding = inputs.get('guidingEnableButton')

    recoverable = {k: v for k, v in safety.get('recoverableErrors', {}).items() if v}
    demanded = {
        k: v
        for k, v in safety.get('demandedRecoveries', {}).items()
        if (any(v) if isinstance(v, list) else v)
    }

    return [
        (
            'operating mode',
            mode == 'Execution',
            mode,
            'Desk must be in Execution mode. Programming/Teach means libfranka rejects every '
            'move command with "User stopped".',
        ),
        (
            'safe torque',
            safety.get('stoState') == 'SafeTorqueOn',
            safety.get('stoState'),
            'Release the user stop button.',
        ),
        (
            'guiding button',
            guiding in (None, 'Inactive'),
            guiding,
            'Let go of the enabling device. Holding it puts the robot in Teach mode, which FCI '
            'cannot drive.',
        ),
        (
            'safety controller',
            controller_status not in ('Teach',),
            controller_status,
            'Teach means hand guiding is active.',
        ),
        (
            'brakes',
            bool(brakes) and all(b == 'Unlocked' for b in brakes),
            'all unlocked' if brakes and all(b == 'Unlocked' for b in brakes) else str(brakes),
            'Unlock the joints in Desk.',
        ),
        (
            'FCI active',
            bool(token.get('fciActive')),
            token.get('fciActive'),
            'Activate FCI in Desk.',
        ),
        (
            'power',
            safety.get('powerState', {}).get('robot') == 'On',
            safety.get('powerState', {}).get('robot'),
            'Robot power is off.',
        ),
        (
            'no recoverable errors',
            not recoverable,
            'none' if not recoverable else ', '.join(recoverable),
            'Acknowledge the errors in Desk.',
        ),
        (
            'no pending recoveries',
            not demanded,
            'none' if not demanded else ', '.join(demanded),
            'Run the demanded recovery in Desk.',
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='robot.example', help='robot address serving Desk')
    parser.add_argument('--timeout', type=float, default=8.0)
    args = parser.parse_args()

    try:
        status = fetch(args.host, args.timeout)
    except Exception as error:  # noqa: BLE001 - any failure here is a plain no-go
        print('cannot reach Desk on %s: %s' % (args.host, error), file=sys.stderr)
        return 2

    results = check(status)
    width = max(len(name) for name, _, _, _ in results)
    print()
    for name, ok, detail, _ in results:
        print('  [%s] %-*s %s' % ('ok' if ok else 'NO', width, name, detail))

    failures = [(name, hint) for name, ok, _, hint in results if not ok]
    light = status.get('derived', {}).get('desiredColor', {}).get('color')
    print('\n  base light: %s' % light)

    if not failures:
        print('\nReady for FCI control.\n')
        return 0

    print('\nNot ready - %d check(s) failed:' % len(failures))
    for name, hint in failures:
        print('  - %s: %s' % (name, hint))
    print()
    return 1


if __name__ == '__main__':
    sys.exit(main())
