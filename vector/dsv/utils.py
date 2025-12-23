import json
import torch

def save_data_to_file(data, filename):
     try:
         with open(filename, 'w') as f:
             json.dump(data, f, indent=4)
         print(f"Results successfully saved to: {data}")
     except Exception as e:
         print(f"Error saving results: {e}")

def load_data_from_file(filename):
    try:
         with open(filename, 'r') as f:
             data = json.load(f)
             return data
    except Exception as e:
         print(f"Error loading results: {e}")

def save_vector_to_file(vector, filename):
    try:
        torch.save(vector, filename)
        print(f"\n--- Final vectors successfully saved to: {filename} ---")

    except Exception as e:
        print(f"\n--- Error saving vectors: {e} ---")

def load_vector_from_file(filename):
    try:
        final_steering_vectors_load = torch.load(filename , weights_only=False)
        print(f"--- Vectors successfully loaded from: {filename} ---")

        # --- 3. (Optional) Verify the loaded data ---
        print("\n--- Example: Layer 15 (from loaded file) ---")
        print(final_steering_vectors_load[15])
        return final_steering_vectors_load

    except Exception as e:
        print(f"\n--- Error loading vectors: {e} ---")
        print("Ensure the file exists and the SteeringVector class is defined.")