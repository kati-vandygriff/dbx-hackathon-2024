import os

def combine_chunks(output_file, chunk_dir, num_chunks):
    with open(output_file, 'wb') as outfile:
        for i in range(num_chunks):
            chunk_file = os.path.join(chunk_dir, f'chunk_{i}.part')
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
            print(f"Added: {chunk_file}")

# Variables to change
output_file = 'helmet_model_reassembled_v1.h5'  # Name of the final reassembled file
chunk_dir = ''  # Directory where the chunks are saved
num_chunks = 22

# Call the combine function
combine_chunks(output_file, chunk_dir, num_chunks)

'''
import os

chunk_dir = ''

# List all chunk files
chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_')])

# Print the size of each chunk
for chunk in chunk_files:
    file_path = os.path.join(chunk_dir, chunk)
    size = os.path.getsize(file_path)
    print(f"{chunk}: {size} bytes")
'''