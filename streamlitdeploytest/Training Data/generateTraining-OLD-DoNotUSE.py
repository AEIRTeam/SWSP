import csv
import random

# Categories and their proportions
labels = {
    "Accident": 1500,
    "Incident": 1500,
    "Near Miss": 1000,
    "Safety Observation": 1000
}

templates = {
    "Accident": [
        "Worker fell from {height} while {task}, resulting in {injury}.",
        "{equipment} malfunction caused {damage}, injuring {victim}.",
        "Exposure to {hazard} led to {injury}.",
        "Worker was struck by {object} during {task}, causing {injury}.",
        "Electrical shock from {equipment} resulted in {injury}.",
        "Chemical burn from {substance} caused {injury} to {victim}.",
        "{victim} drowned when {location} flooded unexpectedly.",
        "Structural collapse of {asset} trapped {victim}, causing {injury}.",
        "Fire caused by {cause} burned {victim}, resulting in {injury}.",
        "Explosion from {hazard} injured {victim} with {injury}.",
        "{victim} suffered {injury} when {object} fell from {height}.",
        "Confined space accident involving {hazard} led to {injury}.",
        "Crush injury occurred when {equipment} pinned {victim} against {object}."
    ],
    "Incident": [
        "{equipment} failure caused {damage} to {asset}.",
        "{substance} spill contaminated {area}.",
        "Structural collapse in {location} due to {cause}.",
        "Uncontrolled flooding in {location} damaged {asset}.",
        "Fire broke out in {area} due to {cause}.",
        "Electrical short circuit in {equipment} caused {damage}.",
        "Gas leak from {object} created hazardous conditions in {area}.",
        "{equipment} collision with {object} resulted in {damage}.",
        "Improperly secured vessel shifted in dry dock, causing {damage}.",
        "High winds caused {object} to break loose, damaging {asset}.",
        "System failure during docking procedure led to {damage}.",
        "Improper maintenance of {equipment} resulted in {damage}."
    ],
    "Near Miss": [
        "Worker nearly {action} but avoided injury.",
        "Unsecured {object} almost fell from {height}.",
        "{hazard} was discovered before harm occurred.",
        "Worker almost stepped into {hazard} while {task}.",
        "{object} nearly struck {victim} during {task}.",
        "Near-electrocution occurred when worker almost touched {hazard}.",
        "Gas leak was detected in {area} before ignition could occur.",
        "Worker almost fell into {location} due to missing guardrail.",
        "Improperly secured load nearly fell from {equipment}.",
        "Worker nearly inhaled toxic fumes from {substance}.",
        "Near-collision between {equipment} and {object} was avoided.",
        "Worker almost crushed between {object} and {asset}."
    ],
    "Safety Observation": [
        "All {equipment} was used correctly during {task}.",
        "{team} followed {protocol} without violations.",
        "Negative: {condition} was observed but not corrected.",
        "Positive: Proper lockout/tagout procedures followed for {equipment}.",
        "Negative: {hazard} was present in {area} without proper signage.",
        "Positive: All workers in {area} were wearing appropriate PPE.",
        "Negative: {object} was improperly stored in {location}.",
        "Positive: Emergency eyewash station in {area} was fully functional.",
        "Negative: Slippery surface in {location} was not marked.",
        "Positive: Fire extinguishers in {area} were properly maintained.",
        "Negative: {equipment} was operated without proper training.",
        "Positive: Safety briefing before {task} covered all hazards."
    ]
}

fillers = {
    "height": ["scaffolding", "ladder", "ship deck", "platform", "gangway", 
              "crane platform", "ship's railing", "elevated walkway", "dock edge",
              "catwalk", "mast", "boom lift"],
    
    "task": ["welding", "painting", "repairing the hull", "inspecting valves", 
            "installing pipes", "operating crane", "loading cargo", "securing lines",
            "cleaning tanks", "grinding metal", "testing systems", "repairing engines",
            "handling chemicals", "working in confined spaces", "operating forklift"],
    
    "injury": ["a fractured leg", "burns", "concussion", "lacerations", "broken arm",
              "spinal injury", "internal bleeding", "respiratory distress", "amputation",
              "crush injuries", "head trauma", "chemical burns", "electric shock",
              "drowning", "hearing loss", "vision impairment", "multiple fractures"],
    
    "equipment": ["crane", "forklift", "grinder", "welding torch", "hoist", 
                 "hydraulic press", "pump system", "electrical panel", "compressor",
                 "winch", "jackhammer", "sander", "cutting torch", "pressure washer",
                 "generator", "conveyor belt", "drill press"],
    
    "damage": ["a steel beam to drop", "a fuel leak", "electrical fire", "structural collapse",
              "pipe rupture", "hull breach", "equipment failure", "system shutdown",
              "toxic release", "flooding", "explosion", "crane collapse", "scaffolding failure",
              "fire outbreak", "chemical spill", "gas leak"],
    
    "victim": ["the operator", "two workers", "a contractor", "a welder", "a painter",
              "a supervisor", "the crane operator", "three crew members", "a technician",
              "an electrician", "a ship fitter", "a rigger", "a safety inspector"],
    
    "hazard": ["toxic fumes", "asbestos", "high-voltage wiring", "flammable liquids",
              "compressed gases", "falling objects", "unguarded machinery", "slippery surfaces",
              "confined space", "oxygen deficiency", "hydrogen sulfide", "ammonia leak",
              "radioactive materials", "noise exposure", "vibration hazards"],
    
    "asset": ["dry dock floor", "ship propeller", "storage area", "engine room",
             "hull plating", "electrical systems", "piping network", "cargo hold",
             "navigation equipment", "lifeboat station", "gangway", "mooring system",
             "ventilation system", "fire suppression system"],
    
    "substance": ["Oil", "Chemicals", "Hydraulic fluid", "Paint thinner", "Solvents",
                 "Acids", "Bases", "Fuel oil", "Compressed gas", "Cleaning agents",
                 "Anti-fouling paint", "Welding fumes", "Exhaust fumes", "Bilge water"],
    
    "area": ["dock area", "engine room", "paint shop", "cargo hold", "machinery space",
            "electrical room", "bow thruster compartment", "stern tube area",
            "accommodation block", "galley", "pump room", "ballast tank"],
    
    "location": ["dry dock", "shipyard warehouse", "pier", "fabrication shop",
                "quayside", "mooring area", "repair bay", "blasting area",
                "coating shed", "outfitting berth", "launching area"],
    
    "cause": ["overload", "corrosion", "improper maintenance", "human error",
             "equipment failure", "high winds", "rough seas", "miscommunication",
             "fatigue", "inadequate training", "poor supervision", "design flaw",
             "material defect", "procedural violation", "weather conditions"],
    
    "action": ["slipped on an oily surface", "touched live wiring", "stepped into an open hatch",
              "fell through unguarded opening", "was nearly crushed by moving equipment",
              "almost inhaled toxic gas", "nearly dropped heavy object", "almost caught in machinery",
              "nearly struck by swinging load", "almost stepped on sharp object"],
    
    "object": ["ladder", "toolbox", "pipe", "steel plate", "cable", "valve",
              "grinding wheel", "chain", "hook", "beam", "vent", "hatch cover",
              "fire extinguisher", "safety rail", "pump"],
    
    "condition": ["untidy cables", "missing guardrails", "blocked emergency exit",
                 "improper PPE usage", "lack of safety signage", "poor housekeeping",
                 "inadequate lighting", "slippery surfaces", "cluttered walkways",
                 "unlabeled chemicals", "expired fire equipment", "broken safety devices"],
    
    "team": ["Paint crew", "Welding team", "Scaffolders", "Electricians", "Riggers",
            "Fitters", "Crane operators", "Safety officers", "Maintenance crew",
            "Dock workers", "Shipwrights", "Engineers"],
    
    "protocol": ["lockout/tagout", "PPE guidelines", "confined space entry",
                "hot work permits", "fall protection", "hazard communication",
                "emergency response", "crane operation", "chemical handling",
                "fire prevention", "first aid procedures", "spill containment"]
}

def generate_entry(label, generated_entries):
    max_attempts = 100  # Failsafe to prevent infinite loops
    for _ in range(max_attempts):
        template = random.choice(templates[label])
        entry = template.format(**{k: random.choice(v) for k, v in fillers.items() if k in template})
        if entry not in generated_entries:
            generated_entries.add(entry)
            return entry
    raise ValueError(f"Could not generate unique entry for label '{label}' after {max_attempts} attempts.")

# Generate CSV
with open('shipyard_hse_dataset_unique.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['incident_description', 'label'])
    generated_entries = set()  # Track all generated entries
    for label, count in labels.items():
        entries_generated = 0
        while entries_generated < count:
            try:
                entry = generate_entry(label, generated_entries)
                writer.writerow([entry, label])
                entries_generated += 1
            except ValueError as e:
                print(e)
                break  # Skip if uniqueness cannot be guaranteed