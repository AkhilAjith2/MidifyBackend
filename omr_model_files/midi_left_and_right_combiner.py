import pretty_midi  # type: ignore


def merge_midi_files(midi_file1, midi_file2, output_file):
    # Load the two MIDI files
    midi1 = pretty_midi.PrettyMIDI(midi_file1)
    midi2 = pretty_midi.PrettyMIDI(midi_file2)

    merged_midi = pretty_midi.PrettyMIDI()

    # Append all instruments from both MIDI files to the merged MIDI
    for instrument in midi1.instruments:
        merged_midi.instruments.append(instrument)

    for instrument in midi2.instruments:
        merged_midi.instruments.append(instrument)

    # Save the merged MIDI file
    merged_midi.write(output_file)
    print(f"Merged MIDI file created successfully: {output_file}")

def merge_multiple_midi_files(midi_files, output_file):
    """ Merges multiple MIDI files sequentially (one after another)."""

    merged_midi = pretty_midi.PrettyMIDI()

    # Track the current offset to place each MIDI file sequentially
    current_time_offset = 0.0

    for midi_file in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(midi_file)

            # Adjust the start time of all notes and append instruments
            for instrument in midi.instruments:
                for note in instrument.notes:
                    note.start += current_time_offset
                    note.end += current_time_offset
                merged_midi.instruments.append(instrument)

            # Update the time offset based on the duration of the current MIDI file
            current_time_offset += midi.get_end_time()

        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            continue

    # Save the merged MIDI file
    merged_midi.write(output_file)
    print(f"Merged MIDI file created successfully: {output_file}")
    return output_file
