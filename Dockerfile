FROM blinkmimic:q1-specialist-runtime-20261110-v14
WORKDIR /workspace/blinkmorphmimic
ENV PYTHONPATH=/workspace/blinkmorphmimic/src MUJOCO_GL=egl OMP_NUM_THREADS=4 BLINKMORPH_DATA_ROOT=/data/kk/data PYTHONDONTWRITEBYTECODE=1
COPY src ./src
COPY preview ./preview
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY LICENSE MORPHMIMIC_LICENSE THIRD_PARTY_NOTICES.md README.md pyproject.toml ./
ENTRYPOINT []
CMD ["python", "-m", "morphmimic.scripts.train", "--help"]
