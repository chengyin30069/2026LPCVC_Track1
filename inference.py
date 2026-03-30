import qai_hub
from utils.track1_utils import evaluate_track1

def run_inference(model, device, input_dataset):
    """Submits an inference job for the model and returns the output data."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1"
    )
    # return inference_job.download_output_data()
    inference_job.wait()
    return inference_job.job_id

#Define target device
device = qai_hub.Device("XR2 Gen 2 (Proxy)")



# TODO: Define tasks with their corresponding compiled job IDs and dataset IDs
tasks = {
    "text": {
        "compiled_id": "j5wdqelmg",
        "dataset_id": "d9pgeq4n9"
    },
    "image": {
        "compiled_id": "jgzxr3lx5",
        "dataset_id": "d7d45ed87"
    }
}

# Dictionary to store outputs separately
outputs = {}

for task_name, info in tasks.items():
    compiled_id = info["compiled_id"]
    input_dataset = qai_hub.get_dataset(info["dataset_id"])

    # Retrieve the compiled model
    job = qai_hub.get_job(compiled_id)
    compiled_model = job.get_target_model()

    # Run inference
    print(f"Running inference for {task_name} model {compiled_model.model_id} on device {device.name}")
    inference_id = run_inference(compiled_model, device, input_dataset)
    inference_job = qai_hub.get_job(inference_id)

    if inference_job.get_status().failure:
        print(f"{task_name.capitalize()} inference failed")
        outputs[task_name] = None
    else:
        inference_output = inference_job.download_output_data()
        outputs[task_name] = inference_output['output_0']

text_output = outputs["text"]
image_output = outputs["image"]

result = evaluate_track1(image_output, text_output, "dataset/txt_list.csv", "dataset/img_list.csv")
print(result)