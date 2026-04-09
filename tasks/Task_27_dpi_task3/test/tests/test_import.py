try:
    from torchkbnufft import KbNufft, AdjKbNufft
    print("KbNufft imported successfully")
except Exception as e:
    print("Error importing KbNufft:", e)
    import torchkbnufft
    print("Dir torchkbnufft:", dir(torchkbnufft))
