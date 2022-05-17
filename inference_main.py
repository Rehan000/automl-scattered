import os
import time


# os.system("pip install -r efficientdet/requirements.txt")
MODEL = 'efficientdet-d3'  #@param

# os.system("pip install motpy")
# os.system("pip install redis")

def download(m):
  if m not in os.listdir():
    if os.path.exists(f"{m}.tar.gz"):
      os.remove(f"{m}.tar.gz")
    os.system(f"wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/{m}.tar.gz")
    os.system(f"tar zxf {m}.tar.gz")
  ckpt_path = os.path.join(os.getcwd(), m)
  return ckpt_path


# Download checkpoint.
min_score_thresh = 0.35  #@param
max_boxes_to_draw = 200  #@param
line_thickness = 2 #@param
os.chdir("efficientdet")
# ckpt_path = os.path.join(os.getcwd(), MODEL)
# os.chdir("..")
# print(ckpt_path)
ckpt_path = download(MODEL)
print("checkpoint_path", ckpt_path)
saved_model_dir = 'savedmodel'
# os.system(f"rm -rf {saved_model_dir}")
print("pwd", os.getcwd())
os.system(f"python model_inspect.py --runmode=saved_model --model_name={MODEL} \
  --ckpt_path={ckpt_path} --saved_model_dir={saved_model_dir}")
serve_image_out = 'serve_image_out'

os.makedirs(serve_image_out, exist_ok=True)

os.system(f"python model_inspect.py --runmode=rstp_infer \
  --saved_model_dir={saved_model_dir} \
  --model_name={MODEL}  --input_image=testdata/img1.jpg  \
  --output_image_dir={serve_image_out} \
  --min_score_thresh={min_score_thresh}  --max_boxes_to_draw={max_boxes_to_draw}")
