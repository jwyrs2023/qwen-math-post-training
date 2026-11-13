"""CPU-safe project entry point; training integrations are pending."""
import argparse
import importlib.metadata
import json
import platform
import sys


def doctor():
    packages = {}
    for name in ('torch', 'transformers', 'datasets', 'accelerate', 'peft', 'trl', 'safetensors', 'PyYAML', 'pytest'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': packages,
        'training_integration_verified': False,
    }
    if packages['torch']:
        import torch
        result['cuda_available'] = torch.cuda.is_available()
        result['cuda_version'] = torch.version.cuda
        result['gpus'] = [
            {'name': torch.cuda.get_device_name(i),
             'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())
        ]
    else:
        result['cuda_available'] = False
        result['gpus'] = []
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='Inspect actual local environment (no model download)')
    d.add_argument('--config')
    prepare = sub.add_parser('prepare-data')
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--output', default='data')
    prepare.add_argument('--revision')
    audit = sub.add_parser('audit-data')
    audit.add_argument('--manifest', default='data/manifests/splits.json')
    for name in ('train-sft', 'train-grpo'):
        train = sub.add_parser(name)
        train.add_argument('--config', required=True)
        train.add_argument('--run-id', required=True)
        train.add_argument('--sft')
        train.add_argument('--dry-run', action='store_true')
        train.add_argument('--max-steps', type=int)
        train.add_argument('--resume')
    freeze = sub.add_parser('freeze-sft')
    freeze.add_argument('--run', required=True)
    freeze.add_argument('--output', required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--config', required=True)
    ev.add_argument('--models', required=True)
    ev.add_argument('--split', choices=['validation', 'official_test'], required=True)
    ev.add_argument('--output', required=True)
    ev.add_argument('--limit', type=int)
    ev.add_argument('--protocol')
    fp = sub.add_parser('freeze-protocol')
    fp.add_argument('--config', required=True)
    fp.add_argument('--models', required=True)
    fp.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'freeze-protocol':
        from .frozen_protocol import freeze_protocol
        from .config import load_config
        freeze_protocol(load_config(args.config), args.models, args.output)
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        from .config import load_config
        print(json.dumps(evaluate(load_config(args.config), args.models, args.split, args.output, args.limit, args.protocol), indent=2))
    elif args.command in ('train-sft', 'train-grpo'):
        from .config import load_config
        from .training import execute
        config = load_config(args.config)
        if args.max_steps is not None:
            if args.max_steps <= 0:
                parser.error('--max-steps must be positive')
            config['max_steps'] = args.max_steps
        print(json.dumps(execute(config, args.run_id, args.command.split('-')[1],
                                 args.sft, args.dry_run, args.resume), indent=2))
    elif args.command == 'freeze-sft':
        from pathlib import Path
        from .checkpoints import freeze
        result = json.loads((Path(args.run) / 'result.json').read_text())
        if result.get('status') != 'complete':
            raise ValueError('cannot freeze an incomplete training run')
        print(json.dumps(freeze(Path(args.run) / 'final', args.output), indent=2))
    elif args.command == 'prepare-data':
        from .data import prepare
        print(prepare(args.output, args.seed, args.revision))
    elif args.command == 'audit-data':
        from .data import audit_manifest
        result = audit_manifest(args.manifest)
        print(json.dumps(result, indent=2))
        if not result['passed']:
            raise SystemExit(2)
    elif args.command == 'doctor':
        if args.config:
            from .config import load_config
            print(json.dumps({'resolved_config': load_config(args.config)}, indent=2))
        doctor()


if __name__ == '__main__':
    main()
