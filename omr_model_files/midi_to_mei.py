from lxml import etree
import subprocess

def convert_midi_to_mei_with_musescore(midi_file, mei_output_file):
    # Path to the MuseScore executable
    musescore_executable = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"  # Update this to the actual path if necessary

    # Command to convert MIDI to MEI using MuseScore
    command = [
        musescore_executable,
        "-o", mei_output_file,  # Output file
        midi_file              # Input file
    ]

    try:
        # Run the command
        subprocess.run(command, check=True)
        print(f"MEI file successfully created: {mei_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during MuseScore conversion: {e}")
    except FileNotFoundError:
        print("MuseScore executable not found. Ensure it is installed and in your PATH.")

def merge_mei_files(left_hand_file, right_hand_file, time_signature, output_file):
    ns = {'mei': 'http://www.music-encoding.org/ns/mei'}

    # Parse the left and right-hand MEI files
    left_tree = etree.parse(left_hand_file)
    right_tree = etree.parse(right_hand_file)

    # Get the root elements
    left_root = left_tree.getroot()
    right_root = right_tree.getroot()

    # Locate <scoreDef> and <section> in both files
    left_score = left_root.find(".//mei:score", namespaces=ns)
    right_score = right_root.find(".//mei:score", namespaces=ns)

    if left_score is None or right_score is None:
        raise ValueError("Both MEI files must have a <score> element.")

    left_section = left_score.find(".//mei:section", namespaces=ns)
    right_section = right_score.find(".//mei:section", namespaces=ns)

    if left_section is None or right_section is None:
        raise ValueError("Both MEI files must have a <section> element.")

    # Add the <staffDef> elements for the merged score
    left_staff_def = left_score.find(".//mei:staffDef", namespaces=ns)
    right_staff_def = right_score.find(".//mei:staffDef", namespaces=ns)

    # Parse the time signature
    try:
        numerator, denominator = map(int, time_signature.split("/"))
    except ValueError:
        raise ValueError("Invalid time signature format. Use 'numerator/denominator' (e.g., '3/4').")

    # Update the staff numbers, labels, and time signature
    if left_staff_def is not None:
        left_staff_def.set("n", "1")
        left_staff_def.set("meter.count", str(numerator))
        left_staff_def.set("meter.unit", str(denominator))

    if right_staff_def is not None:
        right_staff_def.set("n", "2")
        right_staff_def.set("meter.count", str(numerator))
        right_staff_def.set("meter.unit", str(denominator))

    # Create a new <staffGrp> and add the <staffDef> elements in the desired order
    staff_grp = etree.Element("{http://www.music-encoding.org/ns/mei}staffGrp")
    if right_staff_def is not None:
        staff_grp.append(right_staff_def)
    if left_staff_def is not None:
        staff_grp.append(left_staff_def)

    # Insert the new <staffGrp> into the left-hand <scoreDef>
    left_score_def = left_score.find(".//mei:scoreDef", namespaces=ns)
    if left_score_def is not None:
        # Ensure no duplicate <staffGrp> is added
        existing_staff_grp = left_score_def.find(".//mei:staffGrp", namespaces=ns)
        if existing_staff_grp is not None:
            left_score_def.remove(existing_staff_grp)
        left_score_def.insert(0, staff_grp)

    # Remove <label> and <labelAbbr> elements from <staffDef>
    for staff_def in [left_staff_def, right_staff_def]:
        if staff_def is not None:
            label = staff_def.find(".//mei:label", namespaces=ns)
            label_abbr = staff_def.find(".//mei:labelAbbr", namespaces=ns)
            if label is not None:
                staff_def.remove(label)
            if label_abbr is not None:
                staff_def.remove(label_abbr)

    # Create a new <section> for the combined measures
    combined_section = etree.Element("{http://www.music-encoding.org/ns/mei}section")

    # Iterate through measures and merge them
    left_measures = left_section.findall(".//mei:measure", namespaces=ns)
    right_measures = right_section.findall(".//mei:measure", namespaces=ns)

    for left_measure, right_measure in zip(left_measures, right_measures):
        # Create a new combined measure
        measure_number = left_measure.get("n")
        combined_measure = etree.Element("{http://www.music-encoding.org/ns/mei}measure", attrib={"n": measure_number})

        # Append left-hand staff with n="1"
        left_staff = left_measure.find(".//mei:staff", namespaces=ns)
        if left_staff is not None:
            left_staff.set("n", "1")
            combined_measure.append(left_staff)

        # Append right-hand staff with n="2"
        right_staff = right_measure.find(".//mei:staff", namespaces=ns)
        if right_staff is not None:
            right_staff.set("n", "2")
            combined_measure.append(right_staff)

        # Add the combined measure to the new section
        combined_section.append(combined_measure)

    # Replace the original left-hand section with the combined section
    left_score.remove(left_section)
    left_score.append(combined_section)

    # Save the merged MEI file
    with open(output_file, "wb") as output:
        output.write(etree.tostring(left_root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))

    print(f"Merged MEI file saved to: {output_file}")