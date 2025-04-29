import re

# Loop through Day-1.txt to Day-16.txt
for day in range(1, 17):
    file_path = f"Day-{day}.txt"
    
    # Read file content
    with open(file_path, "r") as f:
        content = f.read()
    
    # Replace all standalone "0" with "1"
    updated_content = re.sub(r'\b0\b', '1', content)
    
    # Write changes back to the same file
    with open(file_path, "w") as f:
        f.write(updated_content)

    print(f"Updated {file_path}")