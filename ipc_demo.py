import multiprocessing
import time
import os

def document_worker(task_queue, result_queue):
    """
    Worker process that performs Inter-Process Communication (IPC).
    It reads tasks from task_queue and puts results into result_queue.
    """
    print(f"[Worker] Process ID: {os.getpid()} started.")
    while True:
        task = task_queue.get()
        if task == "STOP":
            print("[Worker] Received STOP signal. Exiting.")
            break
        
        file_path = task
        print(f"[Worker] Processing document: {file_path}")
        
        # Simulate processing time (e.g., embedding generation, chunking)
        time.sleep(1.5)
        
        # In a real scenario, we might call src.ingest.ingest_document(file_path)
        # Here we simulate the IPC response
        if os.path.exists(file_path):
            status = f"SUCCESS: {file_path} processed successfully."
        else:
            status = f"ERROR: File {file_path} not found."
            
        result_queue.put({"file": file_path, "status": status})

def run_ipc_demo():
    print(f"[Main] Process ID: {os.getpid()} started.")
    
    # 1. Create Queues for IPC
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    
    # 2. Start the worker process
    worker = multiprocessing.Process(target=document_worker, args=(task_queue, result_queue))
    worker.start()
    
    # 3. Send tasks to the worker via IPC Queue
    documents_to_process = [
        "./data/Assignment 1.pdf",
        "./data/krrish_resume_new.pdf",
        "./data/Experiment 7.1.pdf",
        "./data/non_existent.pdf"
    ]
    
    print("[Main] Sending tasks to worker via task_queue...")
    for doc in documents_to_process:
        task_queue.put(doc)
        
    # 4. Receive results from the worker via IPC Queue
    processed_count = 0
    while processed_count < len(documents_to_process):
        result = result_queue.get()
        print(f"[Main] Received result from worker: {result['status']}")
        processed_count += 1
        
    # 5. Send STOP signal
    task_queue.put("STOP")
    worker.join()
    print("[Main] Worker process joined. IPC demonstration complete.")

if __name__ == "__main__":
    run_ipc_demo()
