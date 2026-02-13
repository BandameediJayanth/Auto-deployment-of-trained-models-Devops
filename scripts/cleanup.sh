#!/bin/bash
# Cleanup Script for GitHub Repository
# Removes temporary files, logs, and cache before pushing to GitHub

echo "Cleaning up codebase for GitHub..."

# Remove log files
echo "Removing log files..."
find . -name "*.log" -type f -delete 2>/dev/null

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -type f -delete 2>/dev/null

# Remove test results (keep structure)
echo "Cleaning test results..."
find models -name "test_results_*.json" -type f -delete 2>/dev/null

# Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -type f -delete 2>/dev/null
find . -name "*.temp" -type f -delete 2>/dev/null

# Remove OS-specific files
echo "Removing OS-specific files..."
find . -name ".DS_Store" -type f -delete 2>/dev/null
find . -name "Thumbs.db" -type f -delete 2>/dev/null

# Create .gitkeep files for important directories
echo "Creating .gitkeep files..."
for dir in models data logs reports input_models; do
    if [ -d "$dir" ]; then
        touch "$dir/.gitkeep" 2>/dev/null
    fi
done

echo "Cleanup completed!"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'Initial commit: MLOps project'"
echo "4. Push: git push origin main"
