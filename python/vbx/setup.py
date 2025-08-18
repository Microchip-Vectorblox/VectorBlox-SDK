import setuptools


setuptools.setup(
    name="vbx",
    version="2.0.0",
    author="Joe Edwards",
    author_email="joe.edwards@microchip.com",
    description="Python wrapper for vbx cnn simulator",
    install_requires=["onnx==1.16.1",
                      "onnxruntime==1.18.1",
                      "tqdm",
                      "opencv-python==4.7.*"],
    entry_points={'console_scripts':
                  ['vnnx_compile=vbx.generate.lite_flow:main',
                   'vnnx_infer=vbx.sim.__main__:main',
                   'tflite_quantize=vbx.generate.generate_tflite:main',
                   'tflite_infer=vbx.generate.infer_tflite:main',
                   'tflite_compare=vbx.generate.infer_tflite:compare',
                   'tflite_cut=vbx.generate.split_tflite:cut',
                   'tflite_preprocess=vbx.generate.transform_tflite:preprocess',
                   'tflite_postprocess=vbx.generate.transform_tflite:postprocess',
                   'tflite_transform=vbx.generate.transform_tflite:main',
                   'tflite_split=vbx.generate.split_tflite:main',
                   'generate_vbx3_model=vbx.generate.generate_vbx3_model:main',
                   'generate_npy=vbx.generate.generate_npy:main']},
    packages=setuptools.find_packages(),
    package_data={"":["libvbx_cnn_sim.so", "vnnx-types.h"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    python_requires='>=3',
)
 
