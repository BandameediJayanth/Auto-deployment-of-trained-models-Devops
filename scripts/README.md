# Scripts Directory

This directory contains utility scripts for project maintenance and cleanup.

## Available Scripts

### cleanup.ps1 (Windows)
PowerShell script to clean up temporary files, logs, and caches.

**Usage:**
```powershell
.\scripts\cleanup.ps1
```

**What it cleans:**
- Log files (*.log)
- Python cache (__pycache__, *.pyc)
- Temporary files
- Docker volumes (optional)

### cleanup.sh (Linux/Mac)
Bash script to clean up temporary files, logs, and caches.

**Usage:**
```bash
chmod +x scripts/cleanup.sh
./scripts/cleanup.sh
```

**What it cleans:**
- Log files (*.log)
- Python cache (__pycache__, *.pyc)
- Temporary files
- Docker volumes (optional)

### check_model.ps1
PowerShell script to verify model files exist and are valid.

**Usage:**
```powershell
.\scripts\check_model.ps1
```

**Checks:**
- Model file exists
- Metadata file exists
- Model can be loaded
- Model version information

## Creating New Scripts

When adding new scripts:
1. Place them in this directory
2. Add execute permissions (Linux/Mac): `chmod +x script.sh`
3. Document usage in this README
4. Follow naming convention: `action_description.ext`

## Best Practices

- Test scripts in a safe environment first
- Add error handling and validation
- Provide clear output messages
- Document any dependencies
- Use version control for script changes
