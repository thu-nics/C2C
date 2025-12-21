import re
import matplotlib.pyplot as plt
import os

def parse_log_and_plot(log_path, output_image_path):

    current_checkpoint = None
    
    # Regular expressions
    checkpoint_pattern = re.compile(r"checkpoint=(checkpoint-\d+|final)")
    rosetta_pattern = re.compile(r"Rosetta Consistency Rate: ([\d\.]+)")
    slm_pattern = re.compile(r"SLM \(Base\) Consistency Rate: ([\d\.]+)")

    if not os.path.exists(log_path):
        print(f"Error: File not found at {log_path}")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Temporarily store data for each block to keep the mapping correct
    temp_data = {} 

    for line in lines:
        # Find checkpoint
        cp_match = checkpoint_pattern.search(line)
        if cp_match:
            current_checkpoint = cp_match.group(1)
            # Initialize data for this checkpoint
            if current_checkpoint not in temp_data:
                temp_data[current_checkpoint] = {}
            continue

        if current_checkpoint:
            # Find Rosetta consistency rate
            r_match = rosetta_pattern.search(line)
            if r_match:
                temp_data[current_checkpoint]['rosetta'] = float(r_match.group(1))

            # Find SLM consistency rate
            s_match = slm_pattern.search(line)
            if s_match:
                temp_data[current_checkpoint]['slm'] = float(s_match.group(1))

    # Prepare data for plotting (keep order)
    # We sort by checkpoint step; logs typically follow training order.
    
    # Simple sorting logic: extract number; put "final" at the end
    def sort_key(cp_name):
        if cp_name == 'final':
            return float('inf')
        match = re.search(r'(\d+)', cp_name)
        if match:
            return int(match.group(1))
        return 0

    sorted_checkpoints = sorted(temp_data.keys(), key=sort_key)

    valid_checkpoints = []
    valid_rosetta = []
    valid_slm = []

    for cp in sorted_checkpoints:
        data = temp_data[cp]
        if 'rosetta' in data and 'slm' in data:
            valid_checkpoints.append(cp)
            valid_rosetta.append(data['rosetta'])
            valid_slm.append(data['slm'])

    if not valid_checkpoints:
        print("No valid data found to plot.")
        return

    print("Found data:")
    for i, cp in enumerate(valid_checkpoints):
        print(f"{cp}: Rosetta={valid_rosetta[i]}, SLM={valid_slm[i]}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(valid_checkpoints, valid_rosetta, marker='o', label='Rosetta Consistency Rate', linestyle='-', color='b')
    plt.plot(valid_checkpoints, valid_slm, marker='x', label='SLM (Base) Consistency Rate', linestyle='--', color='r')

    plt.title('Consistency Rate vs Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('Consistency Rate')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_image_path)
    print(f"Plot saved to {output_image_path}")

if __name__ == "__main__":
    log_file = "local/checkpoints/qwen3_0.6b+qwen3_32b_include/consistency_output.log"
    output_file = "consistency_plot.png"
    parse_log_and_plot(log_file, output_file)



