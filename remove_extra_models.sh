find torch_uncertainty_models/models/ -type f -name "version_*.safetensors" | while read file; do
  num=$(echo "$file" | grep -oP 'version_\K\d+'); 
  if [ "$num" -gt 50 ]; then 
    ls -f "$file"; 
  fi; 
done
