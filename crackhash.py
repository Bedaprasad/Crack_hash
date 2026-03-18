
"""
# ########################################################

 ██████╗██████╗  █████╗  ██████╗██╗  ██╗    ██╗  ██╗ █████╗ ███████╗██╗  ██╗
██╔════╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝    ██║  ██║██╔══██╗██╔════╝██║  ██║
██║     ██████╔╝███████║██║     █████╔╝     ███████║███████║███████╗███████║
██║     ██╔══██╗██╔══██║██║     ██╔═██╗     ██╔══██║██╔══██║╚════██║██╔══██║
╚██████╗██║  ██║██║  ██║╚██████╗██║  ██╗    ██║  ██║██║  ██║███████║██║  ██║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝

# ########################################################

Brute force hash attack with GPU assist (only for sha1) - Nvidia CUDA only

# ########################################################
# my name is
  
  /$$                       /$$          
| $$                      | $$          
| $$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$ 
| $$__  $$ /$$__  $$ /$$__  $$ |____  $$
| $$  \ $$| $$$$$$$$| $$  | $$  /$$$$$$$
| $$  | $$| $$_____/| $$  | $$ /$$__  $$
| $$$$$$$/|  $$$$$$$|  $$$$$$$|  $$$$$$$
|_______/  \_______/ \_______/ \_______/
                                        
                                        
Examples:
========

==> A
python crackhash.py -a sha1 -p 6dcd4ce23d88e2ee9568ba546c007c63d9131c1b -cpu

==> a
python crackhash.py -a sha1 -p 86f7e437faa5a7fce15d1ddcb9eaeaea377667b8 -cpu

==> Ok
python crackhash.py -a sha1 -p b0a98216a32426b9e66a4ac1eb6df2e96e1b495c -gpu

==> Dry
python crackhash.py -a sha2 -p 702ec3f256e47f56915eb1b7124bf51fef15846f57f0683144168467d9e76523

==> The
python crackhash.py -a md5 -p a4704fd35f0308287f2937ba3eccf5fe

# ########################################################
### This command will take some time to compute but still short
# ########################################################

==> Allo
python crackhash.py -a sha1 -p 2ff43ddd9245873e6ead882d204d7b47d8e54e0e -gpu

==> Goal
python crackhash.py -a md5 -p 12c74214cb4c18cf36d885303d6aa2e1

# ########################################################
### This command may take hours or days to compute
# ########################################################

==> hello
python crackhash.py -a sha1 -p aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d -gpu

"""

# Disable various performance and user warnings related to Numba and CUDA to reduce console clutter
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='numba.cuda.dispatcher')
warnings.filterwarnings('ignore', category=UserWarning, module='numba.cuda.cudadrv.devicearray')
warnings.filterwarnings('ignore', category=UserWarning)  # (optional)

# Standard libraries for hashing, string operations, arrays, CLI parsing, threading, parallelism, etc.
import hashlib
import string
import numpy as np
import argparse
import sys
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Try to import CUDA support from Numba. If unavailable, set 'cuda' to None.
try:
    from numba import cuda
except ImportError:
    cuda = None

# ---------------- Progress Bar Utility Function ----------------

def progress_bar(progress, total, length=40):
    """
    Render a progress bar on the console.
    - progress: Current progress count.
    - total: Total number to reach.
    - length: Length of the bar.
    """
    percent = progress / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # Carriage return to overwrite previous output
    print(f'\rProgress: |{bar}| {progress}/{total} ({percent:.1%})', end='')


    # ---------------- GPU Candidate Generation (if available) ----------------

if cuda:
    @cuda.jit
    def gpu_generate_indices(indices_out, length, charset_len, base):
        """
        CUDA kernel: Compute indices for all possible candidate strings in a batch.
        - indices_out: Output array where each row = a candidate's character indices.
        - length: Length of the candidate strings.
        - charset_len: Size of the charset.
        - base: Offset for this batch.
        """
        idx = cuda.grid(1)
        if idx < indices_out.shape[0]:
            temp = idx + base
            for i in range(length-1, -1, -1):
                indices_out[idx, i] = temp % charset_len
                temp //= charset_len

def candidate_from_indices(indices, charset):
    """
    Convert a numpy array of indices to string candidates using a given charset.
    """
    charset_len = len(charset)
    return [
        ''.join([chr(charset[i % charset_len]) for i in row])
        for row in indices
    ]

def gpu_bruteforce_simple_sha1(target_hash, max_length=5, interrupter=None):
    """
    Brute-force SHA-1 hash on the GPU using candidate generation batches.
    - target_hash: The hash we're trying to crack.
    - max_length: Max password length to test.
    - interrupter: Optional interruption signal (returns True to break).
    """
    charset = string.printable.strip().encode('utf-8')
    charset_len = len(charset)
    found = False

    try:
        for length in range(1, max_length+1):
            total = charset_len ** length                      # Total number of combinations
            batch = 1000000                                    # Size of batch per CUDA call
            print(f"Trying length={length} ({total:,} possibilities)")
            progress = 0
            for base in range(0, total, batch):
                if interrupter and interrupter():
                    print("\n[!] Interrupted by user. Stopping...")
                    return
                current = min(batch, total-base)
                indices_mat = np.zeros((current, length), dtype=np.int32)
                threadsperblock = 128
                blockspergrid = (current + (threadsperblock - 1)) // threadsperblock
                gpu_generate_indices[blockspergrid, threadsperblock](indices_mat, length, charset_len, base)
                cuda.synchronize()  # Wait for GPU to finish batch
                candidates = candidate_from_indices(indices_mat, charset)
                for pwd in candidates:
                    if interrupter and interrupter():
                        print("\n[!] Interrupted by user. Stopping...")
                        return
                    h = hashlib.sha1(pwd.encode()).hexdigest()
                    if h == target_hash:
                        print("\n# ########################################################################")
                        print("# ")
                        print(f"# Found (GPU assist): {pwd}")
                        print("# ")
                        print("# ########################################################################")
                        return
                progress += current
                progress_bar(progress, total)
            print()
        print("Not found (GPU assist)")
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Stopping...")
        

# ---------------- CPU Brute-Force (ThreadPool) ----------------

def calculate_hash(s, algo):
    """
    Compute the hash digest of a string 's' using the specified algorithm.
    Supported: 'md5', 'sha1', 'sha2' ('sha2' means sha256 here).
    """
    if algo == 'md5':
        return hashlib.md5(s.encode()).hexdigest()
    elif algo == 'sha1':
        return hashlib.sha1(s.encode()).hexdigest()
    elif algo == 'sha2':
        return hashlib.sha256(s.encode()).hexdigest()
    else:
        raise ValueError('Unsupported algorithm')
    

def cpu_worker(start_idx, step, algo, target_hash, progress, progress_lock, found_flag, result_holder, stop_event, chars, max_length):
    """
    Worker function for each thread (CPU).
    - start_idx: Thread's unique starting offset.
    - step: Number of threads (stride for each worker).
    - algo: Hash algorithm name.
    - target_hash: Hash to crack.
    - progress/progress_lock: Shared counter/lock for updating progress.
    - found_flag: Shared boolean (in list for mutability).
    - result_holder: Shared list for storing the result if found.
    - stop_event: Shared Event for interrupting all workers.
    - chars: Valid character set.
    - max_length: Max length to try.
    """
    try:
        for length in range(1, max_length+1):
            count = 0
            total = len(chars) ** length
            # itertools.product generates Cartesian product (all combinations)
            for s in itertools.product(chars, repeat=length):
                if stop_event.is_set():
                    return
                if count % step != start_idx:
                    count += 1
                    continue
                test_str = ''.join(s)
                h = calculate_hash(test_str, algo)
                with progress_lock:
                    if not found_flag[0]:
                        progress[0] += 1
                if h == target_hash:
                    with progress_lock:
                        if not found_flag[0]:
                            found_flag[0] = True
                            result_holder[0] = test_str
                            print("\n# ########################################################################")
                            print("# ")
                            print(f"# Result: {test_str}")
                            print("# ")
                            print("# ########################################################################")
                    stop_event.set()
                    return
                count += 1
    except Exception:
        stop_event.set()
        return
    
def cpu_bruteforce(algo, target_hash, max_length=5):
    """
    Brute-force attack on hash using CPU multithreading.
    - algo: Hashing algorithm.
    - target_hash: Hash to crack.
    - max_length: Upper bound on password length.
    """
    max_workers = max(1, (os.cpu_count() or 1) - 1)
    chars = string.printable.strip()
    progress = [0]
    progress_lock = threading.Lock()         # To ensure safe update of progress
    found_flag = [False]                     # Shared mutable flag for breaking out
    result_holder = [None]                   # To share result between threads
    stop_event = threading.Event()           # To signal all threads to stop
    length_progress = {}                     # Placeholder for further stats (unused)

    try:
        for length in range(1, max_length+1):
            total = len(chars) ** length
            print(f"Trying length={length} ({total:,} possibilities)")
            progress[0] = 0
            stop_event.clear()
            futures = []
            # Thread pool for parallel search
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(max_workers):
                    futures.append(executor.submit(
                        cpu_worker, i, max_workers, algo, target_hash,
                        progress, progress_lock, found_flag, result_holder,
                        stop_event, chars, length
                    ))
                try:
                    last_progress = 0
                    while not found_flag[0] and not stop_event.is_set():
                        if progress[0] != last_progress:
                            progress_bar(progress[0], total)
                            last_progress = progress[0]
                        if progress[0] >= total:   # Extra control to break when done
                            stop_event.set()
                            break
                        threading.Event().wait(0.05)
                except KeyboardInterrupt:
                    stop_event.set()
                    print("\n[!] Interrupted by user. Stopping...")
                    break
                # Ensure all workers finish cleanly
                for f in as_completed(futures):
                    pass
            print()
            if found_flag[0]:
                break
        if not found_flag[0] and not stop_event.is_set():
            print("Result: Not found")
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[!] Interrupted by user. Stopping...")

# ---------------- Main Function & CLI Entry ----------------

def main():
    """
    Entry point for the script.
    - Parses command-line arguments.
    - Decides whether to use CPU or GPU brute-force.
    """
    parser = argparse.ArgumentParser(description='Multithreaded Hash Brute Forcer')
    parser.add_argument('-a', choices=['sha1', 'sha2', 'md5'], required=True, help='Hash algorithm')
    parser.add_argument('-p', required=True, help='Hash to crack')
    parser.add_argument('-gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('-cpu', action='store_true', help='Use CPU (default)')
    args = parser.parse_args()

    algo = args.a
    target_hash = args.p
    use_gpu = args.gpu
    use_cpu = args.cpu or (not use_gpu)
    max_length = 5

    options = []
    if use_cpu:
        options.append('-cpu')
    if use_gpu:
        options.append('-gpu')

    print("""
 ██████╗██████╗  █████╗  ██████╗██╗  ██╗    ██╗  ██╗ █████╗ ███████╗██╗  ██╗
██╔════╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝    ██║  ██║██╔══██╗██╔════╝██║  ██║
██║     ██████╔╝███████║██║     █████╔╝     ███████║███████║███████╗███████║
██║     ██╔══██╗██╔══██║██║     ██╔═██╗     ██╔══██║██╔══██║╚════██║██╔══██║
╚██████╗██║  ██║██║  ██║╚██████╗██║  ██╗    ██║  ██║██║  ██║███████║██║  ██║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
                                                  
    """)

    print("# ########################################################################")
    print("# ")
    print(f"# hash: {target_hash}")
    print(f"# algo: {algo}")
    print(f"# options: {' '.join(options)}")
    print("# ")
    print("# ########################################################################\n")


    if use_gpu:
        # Check for CUDA and GPU compatibility
        if cuda is None:
            print("[!] Numba CUDA not installed.")
            sys.exit(1)
        if not cuda.is_available():
            print("[!] GPU requested but no compatible CUDA GPU found.")
            sys.exit(1)
        if algo != 'sha1':
            print("[!] GPU assist mode only for sha1 currently.")
            sys.exit(1)
        # Install interrupter based on signal (KeyboardInterrupt not handled in GPU kernel directly)
        stop_event = threading.Event()
        def interrupted():
            return stop_event.is_set()
        try:
            import signal
            def handler(s, f):
                stop_event.set()
            signal.signal(signal.SIGINT, handler)
            gpu_bruteforce_simple_sha1(target_hash, max_length=max_length, interrupter=interrupted)
        except KeyboardInterrupt:
            stop_event.set()
            print("\n[!] Interrupted by user. Stopping...")
        return

    # ---- CPU fallback search ----
    try:
        cpu_bruteforce(algo, target_hash, max_length=max_length)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Stopping...")

# If script is run directly, start main(). (Supports CLI usage)
if __name__ == '__main__':
    main()
