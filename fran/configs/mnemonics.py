
# %%
from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class Mnemonic:
    name: str
    aliases: tuple[str, ...] = ()

class Mnemonics:
    lungs: ClassVar[Mnemonic] = Mnemonic("lungs", ("lung", "lidc"))
    liver: ClassVar[Mnemonic] = Mnemonic("liver", ("liver", "lits", "litsmc"))
    nodes : ClassVar[Mnemonic] = Mnemonic("nodes", ("nodes"))
    pancreas: ClassVar[Mnemonic] = Mnemonic("pancreas", ("pancreas"))
    colon: ClassVar[Mnemonic] = Mnemonic("colon", ("colon"))
    totalseg : ClassVar[Mnemonic] = Mnemonic("totalseg", ("totalseg"))

    _all: ClassVar[tuple[Mnemonic, ...]] = (lungs, liver, nodes, totalseg, pancreas, colon)

    _index: ClassVar[dict[str, Mnemonic]] = {
        key: m
        for m in _all
        for key in (m.name, *m.aliases)
    }

    @classmethod
    def match(cls, s: str) -> str:
        try:
            return cls._index[s.strip().lower()].name
        except KeyError:
            raise ValueError(f"Unknown mnemonic: {s}")

# %%
