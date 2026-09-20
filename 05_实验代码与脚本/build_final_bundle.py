#!/usr/bin/env python3
"""
Build Final Cross-Domain Bundle
==============================
生成最终的上传包
"""

import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

def main():
    bundle_dir = Path('cloud_4090_cross_domain_bundle')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f'cloud_4090_cross_domain_bundle_{timestamp}.zip'
    
    print('='*60)
    print('BUILDING CROSS-DOMAIN BUNDLE')
    print('='*60)
    print()
    
    # Count files
    all_files = list(bundle_dir.rglob('*'))
    files_to_add = [f for f in all_files if f.is_file()]
    
    print(f'Files to include: {len(files_to_add)}')
    print()
    
    # Create ZIP
    print('Creating ZIP archive...')
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_add:
            arcname = str(file_path.relative_to(bundle_dir))
            zf.write(file_path, arcname)
            size = file_path.stat().st_size
            if size > 1024*1024:
                size_str = f'{size/1024/1024:.1f} MB'
            else:
                size_str = f'{size/1024:.1f} KB'
            print(f'  Added: {arcname:<50} {size_str:>10}')
    
    # Calculate checksum
    print()
    print('Calculating SHA256 checksum...')
    sha256 = hashlib.sha256()
    with open(output_name, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    
    with open(f'{output_name}.sha256', 'w') as f:
        f.write(f'{checksum}  {output_name}\n')
    
    file_size = Path(output_name).stat().st_size / 1024 / 1024
    
    print()
    print('='*60)
    print('BUILD SUCCESSFUL!')
    print('='*60)
    print(f'Output file: {output_name}')
    print(f'File size:   {file_size:.1f} MB')
    print(f'SHA256:      {checksum}')
    print(f'Checksum:    {output_name}.sha256')
    print()
    print('Ready to upload!')
    print()
    print('Upload instructions:')
    print(f'  scp {output_name} user@server:/path/')
    print(f'  # or use FTP/SFTP/FileZilla')

if __name__ == '__main__':
    main()
