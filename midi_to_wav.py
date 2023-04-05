from midi2audio import FluidSynth
import sys

FluidSynth().midi_to_audio(f'{sys.argv[1]}', f'{sys.argv[2]}')
