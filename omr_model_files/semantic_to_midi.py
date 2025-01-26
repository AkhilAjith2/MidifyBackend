import pretty_midi

# Define a mapping for notes
note_mapping = {
    'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4, 'E#': 5, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11
}

# Define note durations in beats
base_durations = {
    'sixty_fourth': 0.0625,
    'thirty_second': 0.125,
    'sixteenth': 0.25,
    'eighth': 0.5,
    'quarter': 1.0,
    'half': 2.0,
    'whole': 4.0
}

# Fixed duration for grace notes
grace_note_duration = 0.1


def get_midi_note_number(note):
    """Convert note string to MIDI note number."""
    pitch, octave = note[:-1], int(note[-1])
    return (octave + 1) * 12 + note_mapping[pitch]


def calculate_duration(note_duration, numerator):
    """Calculate duration in beats."""
    if note_duration == "whole":
        return numerator
    base_duration = base_durations.get(note_duration.replace('.', ''), 1.0)
    if note_duration.endswith('.'):
        base_duration += base_duration / 2
    return base_duration

def semantic_to_midi(semantic_data, output_file, tempo=74, time_signature="4/4"):
    numerator, denominator = map(int, time_signature.split('/'))
    note_value_per_beat = 4 / denominator
    seconds_per_beat = (60 / tempo)
    beats_per_measure = numerator * note_value_per_beat
    seconds_per_measure = beats_per_measure * seconds_per_beat

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
    midi.instruments.append(piano)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(numerator=numerator, denominator=denominator, time=0.0))

    current_time = 0
    measure_accumulated_duration = 0  # Tracks accumulated duration within a measure

    for element in semantic_data:
        if element.startswith('note-') or element.startswith('rest-'):
            note_type, note_info = element.split('-', 1)

            if note_type == 'note':
                note_pitch, note_duration = note_info.split('_', 1)
                if '_fermata' in note_duration:
                    note_duration = note_duration.replace('_fermata', '')

                midi_note = get_midi_note_number(note_pitch)
                duration = calculate_duration(note_duration, numerator) * seconds_per_beat

                # Check if adding the note exceeds the measure duration
                if measure_accumulated_duration + duration > seconds_per_measure:
                    remaining_duration = seconds_per_measure - measure_accumulated_duration
                    if remaining_duration > 0:  # Truncate the note to fit the remaining time
                        end_time = current_time + remaining_duration
                        note = pretty_midi.Note(velocity=100, pitch=midi_note, start=current_time, end=end_time)
                        piano.notes.append(note)
                        current_time = end_time
                    measure_accumulated_duration = seconds_per_measure
                    continue  # Move to the next element since this measure is complete

                # Add the note as is
                end_time = current_time + duration
                note = pretty_midi.Note(velocity=100, pitch=midi_note, start=current_time, end=end_time)
                piano.notes.append(note)
                current_time = end_time
                measure_accumulated_duration += duration

            elif note_type == 'rest':
                rest_duration = calculate_duration(note_info, numerator) * seconds_per_beat

                # Check if adding the rest exceeds the measure duration
                if measure_accumulated_duration + rest_duration > seconds_per_measure:
                    remaining_duration = seconds_per_measure - measure_accumulated_duration
                    if remaining_duration > 0:  # Truncate the rest to fit the remaining time
                        current_time += remaining_duration
                    measure_accumulated_duration = seconds_per_measure
                    continue  # Move to the next element since this measure is complete

                # Add the rest as is
                current_time += rest_duration
                measure_accumulated_duration += rest_duration

        elif element.startswith('barline'):
            # Finalize the current measure
            if measure_accumulated_duration < seconds_per_measure:
                # Fill remaining time with rest
                remaining_duration = seconds_per_measure - measure_accumulated_duration
                current_time += remaining_duration
                print(f"Filling measure with {remaining_duration:.2f} seconds of rest.")
            # Reset measure accumulator for the next measure
            measure_accumulated_duration = 0

    # Write the MIDI file
    midi.write(output_file)
    print(f"MIDI file created successfully: {output_file}")

def remove_extra_metadata(predictions):
    """
    Remove extra occurrences of clefs, time signatures, and key signatures from the prediction data.

    Args:
        predictions (list): List of musical elements in the data.

    Returns:
        list: Cleaned list with only the first occurrences of clefs, time signatures, and key signatures.
    """
    metadata = {"clef", "timeSignature", "keySignature"}
    seen_metadata = set()
    cleaned_predictions = []

    for element in predictions:
        prefix = element.split('-')[0]
        if prefix in metadata:
            if prefix not in seen_metadata:
                cleaned_predictions.append(element)
                seen_metadata.add(prefix)
        else:
            cleaned_predictions.append(element)

    return cleaned_predictions
