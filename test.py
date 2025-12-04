import vllm
print(f'vLLM version: {vllm.__version__}')
import time
from vllm import LLM, SamplingParams

# Initialize with sleep mode enabled
print("🔧 Initializing vLLM with sleep mode...")
llm = LLM(
    model="Qwen/Qwen3-0.6B",  # Small model for quick testing
    enable_sleep_mode=True,
    dtype="bfloat16",
    max_model_len=128,
    max_num_seqs=1,
    max_num_batched_tokens=128,
)

print("✅ Model loaded successfully!")

# Test 1: Generate before sleep
print("\n📝 Test 1: Generating text before sleep...")
prompts = [
    "The future of artificial intelligence is",
    "In the year 2050, technology will",
]
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for testing
    max_tokens=25,
)

for i in range(1):
    outputs_before = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs_before):
        print(f"  Prompt {i+1}: {output.outputs[0].text[:100]}")

# Test 2: Sleep and wake WITH STATE PRESERVATION
print("\n💤 Test 2: Entering sleep mode with state preservation...")
llm.sleep(level=1, preserve_state=True)  # 🔑 preserve_state=True
print("✅ Engine is sleeping with state preserved")

time.sleep(4)

print("\n🌅 Waking up...")
llm.wake_up()
print("✅ Engine is awake")


# Test 3: Generate after wake (should be identical with temp=0)
print("\n📝 Test 3: Generating same prompts after wake...")
outputs_after = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs_after):
    print(f"  Prompt {i+1}: {output.outputs[0].text[:100]}...")

# Verify consistency
print("\n🔍 Verifying consistency...")
for i in range(len(prompts)):
    before = outputs_before[i].outputs[0].text
    after = outputs_after[i].outputs[0].text
    if before == after:
        print(f"  ✅ Prompt {i+1}: Output matches (state preserved correctly!)")
    else:
        print(f"  ❌ Prompt {i+1}: Output differs!")
        print(f"     Before: {before[:50]}...")
        print(f"     After:  {after[:50]}...")

