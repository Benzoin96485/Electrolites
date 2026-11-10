"""Sample one benchmark process's host and device memory from outside Python.

Launch under the same scheduler step and environment as the calculation.
GPU values are process memory reported by nvidia-smi, including allocator
reservations. Their maximum is a sampled lower bound on the true peak.
Host VmHWM is the kernel's high-water mark while the process is alive.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.interval <= 0:
        parser.error('Provide a command and a positive sampling interval')
    path = Path(args.output)
    if path.exists():
        parser.error('Resource report already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    process = subprocess.Popen(command)
    report = dict(complete=False, command=command, pid=process.pid,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  interval_seconds=args.interval, gpu_peak_is_sampled_lower_bound=True,
                  samples=[], errors=[])

    def save():
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)

    save()
    while process.poll() is None:
        sample = dict(seconds=time.monotonic()-start)
        try:
            for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('VmRSS', 'VmHWM'):
                    sample[key+'_kib'] = int(value.split()[0])
        except FileNotFoundError:
            pass
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, check=True)
            sample['gpu_mib'] = {}
            for line in result.stdout.splitlines():
                uuid, pid, memory = (x.strip() for x in line.split(','))
                if int(pid) == process.pid:
                    sample['gpu_mib'][uuid] = int(memory)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report['errors'].append(dict(seconds=sample['seconds'], error=str(error)))
        report['samples'].append(sample)
        report['sampled_peak_gpu_mib'] = max(
            sum(x.get('gpu_mib', {}).values()) for x in report['samples'])
        report['observed_host_hwm_kib'] = max(
            x.get('VmHWM_kib', 0) for x in report['samples'])
        save()
        try:
            process.wait(timeout=args.interval)
        except subprocess.TimeoutExpired:
            pass
    report.update(complete=True, returncode=process.returncode,
                  elapsed_seconds=time.monotonic()-start)
    save()
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
