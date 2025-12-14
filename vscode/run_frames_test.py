import os
import importlib.util

base = os.path.dirname(__file__)
mod_path = os.path.join(base, 'python_content_test_cloude.py')

spec = importlib.util.spec_from_file_location('pc', mod_path)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

# Create a small fast config
cfg = pc.Config()
cfg.width = 480
cfg.height = 854
cfg.fps = 12
cfg.title_font_size = 36
cfg.subtitle_font_size = 20
cfg.code_font_size = 16
cfg.brand_font_size = 18

outputs = os.path.join(base, 'outputs')
os.makedirs(outputs, exist_ok=True)

gen = pc.FrameGen(cfg, pc.THEMES['dark_tech'])

# Save a title frame and a code frame as PNGs for a quick visual smoke test
tf = gen.title_frame('Fast Test', 'Quick frame render test')
tf_path = os.path.join(outputs, 'fast_test_title.png')
tf.save(tf_path)

cf = gen.code_frame('Fast Test', ['import math', 'print(math.gcd(12, 18))'], visible=2)
cf_path = os.path.join(outputs, 'fast_test_code.png')
cf.save(cf_path)

print('Saved frames:')
print(' ', tf_path)
print(' ', cf_path)
import os
import importlib.util

base = os.path.dirname(__file__)
mod_path = os.path.join(base, 'python_content_test_cloude.py')

spec = importlib.util.spec_from_file_location('pc', mod_path)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

# Create a small fast config
cfg = pc.Config()
cfg.width = 480
cfg.height = 854
cfg.fps = 12
cfg.title_font_size = 36
cfg.subtitle_font_size = 20
cfg.code_font_size = 16
cfg.brand_font_size = 18

outputs = os.path.join(base, 'outputs')
os.makedirs(outputs, exist_ok=True)

gen = pc.FrameGen(cfg, pc.THEMES['dark_tech'])

# Save a title frame and a code frame as PNGs for a quick visual smoke test
tf = gen.title_frame('Fast Test', 'Quick frame render test')
tf_path = os.path.join(outputs, 'fast_test_title.png')
tf.save(tf_path)

cf = gen.code_frame('Fast Test', ['import math', 'print(math.gcd(12, 18))'], visible=2)
cf_path = os.path.join(outputs, 'fast_test_code.png')
cf.save(cf_path)

print('Saved frames:')
print(' ', tf_path)
print(' ', cf_path)
