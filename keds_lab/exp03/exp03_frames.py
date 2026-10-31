"""Experiment 3: Frames with defaults, inheritance and if-needed procedures"""


class Frame:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.slots = {}          # slot -> {"value","default","if_needed","if_added"}

    def define(self, slot, value=None, default=None, if_needed=None, if_added=None):
        self.slots[slot] = {"value": value, "default": default,
                            "if_needed": if_needed, "if_added": if_added}

    def get(self, slot, origin=None):
        """Order: own value -> if-needed procedure -> default -> parent frame.
        Returns (value, source). `origin` is the frame the query started from, so an
        inherited if-needed procedure computes from the *asking* frame's slots."""
        origin = origin or self
        s = self.slots.get(slot)
        if s:
            if s["value"] is not None:
                return s["value"], f"frame value ({self.name})"
            if s["if_needed"] is not None:
                return s["if_needed"](origin), f"if-needed procedure ({self.name})"
            if s["default"] is not None:
                return s["default"], f"default ({self.name})"
        if self.parent:
            v, src = self.parent.get(slot, origin)
            return v, (src if src.startswith(("none", "inherited")) else f"inherited: {src}")
        return None, "none"

    def set(self, slot, value):
        s = self.slots.setdefault(slot, {"value": None, "default": None,
                                         "if_needed": None, "if_added": None})
        demon = s["if_added"] or self._inherited_demon(slot)
        if demon:
            demon(self, value)
        s["value"] = value

    def _inherited_demon(self, slot):
        f = self.parent
        while f:
            if slot in f.slots and f.slots[slot]["if_added"]:
                return f.slots[slot]["if_added"]
            f = f.parent
        return None


# --- procedures --------------------------------------------------------------
def compute_mileage(frame):
    odo, _ = frame.get("odometer")
    fuel, _ = frame.get("fuel_used")
    if odo is None or not fuel:
        return None
    return round(odo / fuel, 2)


def warn_negative(frame, value):
    if value < 0:
        print(f"   [DEMON] WARNING: {frame.name}: negative value {value} set!")


# --- hierarchy ---------------------------------------------------------------
vehicle = Frame("Vehicle")
vehicle.define("wheels", default=4)
vehicle.define("fuel_type", default="petrol")
vehicle.define("odometer")
vehicle.define("fuel_used")
vehicle.define("mileage", if_needed=compute_mileage, if_added=warn_negative)

car = Frame("Car", parent=vehicle)
car.define("doors", default=4)
car.define("category", value="passenger")

my_car = Frame("MyCar", parent=car)
my_car.define("colour", value="red")
my_car.define("odometer", value=12000)
my_car.define("fuel_used", value=800)
my_car.define("doors", value=2)

print("=== Slot values of MyCar and their source ===")
print(f"{'Slot':<12}{'Value':<10}Source")
print("-" * 55)
for slot in ["colour", "doors", "category", "wheels", "fuel_type", "odometer", "fuel_used", "mileage"]:
    v, src = my_car.get(slot)
    print(f"{slot:<12}{str(v):<10}{src}")

print("\n=== Demon demonstration ===")
print("Setting mileage = -5 on MyCar:")
my_car.set("mileage", -5)
print("Setting mileage = 15 on MyCar (no warning):")
my_car.set("mileage", 15)
print("mileage now ->", my_car.get("mileage"))
