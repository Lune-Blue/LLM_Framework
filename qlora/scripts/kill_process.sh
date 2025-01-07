lsof -i :29500 | awk 'NR>1 {print $2}' | xargs kill -9
