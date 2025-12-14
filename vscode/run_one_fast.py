import os
import importlib.util

base = os.path.dirname(__file__)
mod_path = os.path.join(base, 'python_content_test_cloude.py')

spec = importlib.util.spec_from_file_location('pc', mod_path)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

# Make a fast, low-res config by adjusting class defaults for a quick test
pc.Config.width = 480
pc.Config.height = 854
pc.Config.fps = 12
pc.Config.duration_per_line = 0.5
pc.Config.title_duration = 1.0
pc.Config.outro_duration = 1.0

outputs = os.path.join(base, 'outputs')
os.makedirs(outputs, exist_ok=True)

out = os.path.join(outputs, '01_test_fast.mp4')
print('Running fast test, output:', out)
pc.generate_video(pc.INPUT_TEXTS[0], out, theme_name='dark_tech')
print('Done')
