from opentrons import protocol_api
import csv
import json

csv_path = "/data/user_storage/prd_protocols/Duplicated_Volumes.csv"
# csv_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\LCST\23-Jan full plate + salt + HCl\Duplicated_Volumes.csv"


# Dictionary to hold volumes for each column
# volumes_dict = {}
# Define fake volumes_dict
volumes_dict = {
    "styrene": [0, 0, 300, 300],
    "polystyrene": [300, 300, 0, 0],
}

# with open(csv_path, mode='r', newline="") as file:
#     csv_reader = csv.DictReader(file)
#
#     # Initialize lists in the dictionary for each column header
#     for column in csv_reader.fieldnames:
#         volumes_dict[column] = []
#
#     # Iterate over rows and populate lists in the dictionary
#     for row in csv_reader:
#         for column, value in row.items():
#             volumes_dict[column].append(float(value))
#
# with open(csv_path, mode='r', newline="") as file:
#     csv_reader = csv.DictReader(file)
#     # Count the number of rows in the CSV file after the header
#     num_samples = sum(1 for _ in csv_reader) // 2

num_samples=2

# Define constants
total_volume = 300  # final volume in each well
step_size = 20  # minimum step size
num_factors = 3  # number of variables (styrene, polystyrene, etc)
well_height = 10.9  # mm from top to bottom of well

# Define metadata for protocol
metadata = {
    "apiLevel": "2.19",
    "protocolName": "DOE Mixtures - SSH",
    "description": """
    From CSV input, produces 46 unique samples of varying polymer and monomer concentrations. The first four wells are
    intended to be used as blanks. 
    """,
    "author": "Lachlan Alexander",
    "date last modified": "23-Oct-2024",
    "change log": "Added SSH-friendly labware loading with offsets taken from OT app. "
                  "Added dynamic volume loading from CSV loaded to OT-2 directory."
                  "Added capability to handle dispensing into plate reader labware."
                  "Added building block commands to work with custom plate defs."
}

# Constants
PIPETTE_R_NAME: str = 'p1000_single_gen2'
PIPETTE_L_NAME: str = 'p300_single_gen2'

R_PIP_TIPRACK_SLOTS: list = [8]
R_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_1000ul'

L_PIP_TIPRACK_SLOTS: list = [9]
L_PIP_TIPRACK_LOADNAME: str = 'opentrons_96_tiprack_300ul'

RESERVOIR_SLOTS: list = [2]
RESERVOIR_LOADNAME: str = 'opentrons_15_tuberack_falcon_15ml_conical'

WELL_PLATE_SLOTS: list = [3]
WELL_PLATE_LOADNAME: str = 'greiner_96_wellplate_300ul'
# WELL_PLATE_LOADNAME: str = 'biorad_96_wellplate_200ul_pcr'  # simulation only, remove for actual protocol

# Begin protocol


def run(protocol: protocol_api.ProtocolContext):
    # Load predefined labware on deck
    r_tipracks: list = [protocol.load_labware(R_PIP_TIPRACK_LOADNAME, slot) for slot in R_PIP_TIPRACK_SLOTS]
    l_tipracks: list = [protocol.load_labware(L_PIP_TIPRACK_LOADNAME, slot) for slot in L_PIP_TIPRACK_SLOTS]
    reservoirs: list = [protocol.load_labware(RESERVOIR_LOADNAME, slot) for slot in RESERVOIR_SLOTS]

    def load_custom_labware(file_path, location):
        with open(file_path) as labware_file:
            labware_def = json.load(labware_file)
        return protocol.load_labware_from_definition(labware_def, location)

    try:
        # Load custom labware on deck
        plates = [load_custom_labware("/data/user_storage/labware/slot 4 working.json", 4),
                  load_custom_labware("/data/user_storage/labware/slot 7 working ordered.json", 7),
                  ]

    except Exception as e:
        # Load definition stored in PC namespace if fails
        plates = [protocol.load_labware("plate_reader_slot_4_ordered", 4),
                  protocol.load_labware("plate_reader_slot_7_ordered", 7)]

    # Load labware offsets (FROM OPENTRONS APP - PLEASE ENSURE THESE ARE FILLED BEFORE RUNNING EXECUTE)
    r_tipracks[0].set_offset(x=0.30, y=-1.20, z=-1.20)
    l_tipracks[0].set_offset(x=0.00, y=0.70, z=0.00)
    reservoirs[0].set_offset(x=-0.40, y=-1.40, z=-0.50)
    plates[0].set_offset(x=1.40, y=-1.30, z=-12.70)  # Please for the love of god make sure these two are set properly
    plates[1].set_offset(x=1.60, y=-1.50, z=-13.10)

    # Load pipettes
    right_pipette = protocol.load_instrument(PIPETTE_R_NAME, "right", tip_racks=r_tipracks)
    left_pipette = protocol.load_instrument(PIPETTE_L_NAME, "left", tip_racks=l_tipracks)

    # Begin liquid handling steps

    # Prepare well positions
    sample_wells = []  # Reset list
    for row1, row2 in zip(plates[0].columns(), plates[1].columns()):
        sample_wells.extend(row1)  # Add wells from the current row of slot 4 definition
        sample_wells.extend(row2)  # Add wells from the current row of slot 7 definition

    # Prepare target wells for experimental samples
    target_wells = [well for pair in zip(sample_wells[::2], sample_wells[1::2]) for well in pair][:2 * num_samples]
    target_wells_bottom = [wll.bottom(well_height / 2.5) for wll in target_wells]
    target_wells_top = [wll.top() for wll in target_wells]

    right_pipette.flow_rate.aspirate = 41.175*4

    # Reverse the order of the components for dispensing
    for index, (component, volumes) in enumerate(volumes_dict.items()):
        protocol.comment(f"Adding {component} to wells")

        # Use a unique well in reservoirs[0] for each component, starting from the last well
        source_well = reservoirs[0].wells()[index]

        # Distribute the volume from the specified source well to target wells
        right_pipette.distribute(
            volume=volumes,
            source=source_well,
            dest=target_wells_top,
            blow_out=True,
            blowout_location="source well",
            air_gap=20
        )

    protocol.comment("Protocol Finished")
