# scripts/backup.py

import os
import shutil
import argparse
from datetime import datetime

def backup_data(source_dir, backup_dir):
    """Backup data from the source directory to the backup directory."""
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
    
    shutil.copytree(source_dir, backup_path)
    print(f"Backup completed successfully! Data backed up to: {backup_path}")

def restore_data(backup_path, restore_dir):
    """Restore data from the backup directory to the restore directory."""
    if os.path.exists(restore_dir):
        shutil.rmtree(restore_dir)  # Remove existing restore directory

    shutil.copytree(backup_path, restore_dir)
    print(f"Restore completed successfully! Data restored to: {restore_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backup and restore script.')
    parser.add_argument('action', choices=['backup', 'restore'], help='Action to perform: backup or restore')
    parser.add_argument('source', help='Source directory for backup or backup directory for restore')
    parser.add_argument('--destination', help='Destination directory for restore', required=False)

    args = parser.parse_args()

    if args.action == 'backup':
        backup_data(args.source, 'backups')
    elif args.action == 'restore' and args.destination:
        restore_data(args.source, args.destination)
    else:
        print("Invalid arguments. Please provide the correct parameters.")
