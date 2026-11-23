/** Waits for the GPU to finish what was submitted, then for Dawn to map the timestamps it read. */
export async function timestampsRead(device: GPUDevice) {
  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 100));
}
