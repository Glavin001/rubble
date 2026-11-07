"""Design inheritance: a design module starts from another design and
overrides only what differs.

    from vehgen.design import derive
    derive('sedan', globals())
    NAME = 'police_sedan'
    PAINT = (...)
    _base_parts = parts
    def parts(cb):
        out, extras, mk = _base_parts(cb)
        ...

derive() runs the base design's source in the derived module's namespace,
so the base's functions (tube(), front_z(), assign(), parts(), ...) read the
derived module's globals: overriding a constant, a key table or a function
changes every base function that uses it. Keep a reference to a base
function (as `_base_parts` above) before redefining it to extend it.
"""
import os

DESIGNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'designs')


def derive(base, ns):
    path = os.path.join(DESIGNS, base + '.py')
    keep = {k: ns[k] for k in ('__name__', '__doc__', '__file__') if k in ns}
    with open(path) as f:
        code = compile(f.read(), path, 'exec')
    exec(code, ns)
    ns.update(keep)
    ns['BASE'] = base
    return ns
