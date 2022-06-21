import setuptools


setuptools.setup(
    name="vbx",
    version="1.3.3",
    author="Joe Edwards",
    author_email="joe.edwards@microchip.com",
    description="Python wrapper for vbx cnn simulator",
    install_requires=["onnx==1.9.0",
                      "onnxruntime==1.8.0",
                      "tqdm",
                      "opencv-python==4.5.*"],
    entry_points={'console_scripts':
                  ['generate_vnnx=vbx.generate.__main__:main',
                   'simulate_vnnx=vbx.sim.__main__:main',
                   'openvino_infer=vbx.generate.openvino_infer:main',
                   'onnx_infer=vbx.generate.onnx_infer:main']},
    packages=setuptools.find_packages(),
    package_data={"":["libvbx_cnn_sim.so", "vnnx-types.h"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    python_requires='>=3',
)
 
