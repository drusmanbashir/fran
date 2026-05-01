# %%
'''
mnemonics workflow:
    1- To retrieve config plan
         - at project_init 
         - config plans 
         both above match  = ConfigMaker gets relevant row
    2- Match with a canonical wandb name.
    3- Aliases ensure that whatever the mnemonic is given at project init, it matches the same wandb name and same config plan row
'''

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Mnemonic:
    name: str
    aliases: tuple[str, ...] 
    wandb: str


class Mnemonics:
    lungs: ClassVar[Mnemonic] = Mnemonic("lungs", ("lung", "lidc"), "LIDC")
    liver: ClassVar[Mnemonic] = Mnemonic("liver", ("liver", "lits", "litsmc"), "LITS")
    kidneys: ClassVar[Mnemonic] = Mnemonic("kidneys", ("kidney", "kits2", "kits23"), "KITS")
    nodes: ClassVar[Mnemonic] = Mnemonic("nodes", ("nodes",), "NODES")
    pancreas: ClassVar[Mnemonic] = Mnemonic("pancreas", ("pancreas",), "PANCREAS")
    colon: ClassVar[Mnemonic] = Mnemonic("colon", ("colon",), "COLON")
    totalseg: ClassVar[Mnemonic] = Mnemonic("totalseg", ("totalseg",), "TOTALSEG")
    test: ClassVar[Mnemonic] = Mnemonic("test", ("test",), "TEST")

    _all: ClassVar[tuple[Mnemonic, ...]] = (
        lungs,
        liver,
        kidneys,
        nodes,
        totalseg,
        pancreas,
        colon,
        test,
    )

    _index: ClassVar[dict[str, Mnemonic]] = {
        key: m for m in _all for key in (m.name, *m.aliases)
    }

    @classmethod
    def match(cls, s: str) -> str:
        try:
            return cls._index[s.strip().lower()].name
        except KeyError:
            raise ValueError(f"Unknown mnemonic: {s}")

    def __getitem__(self, s: str) -> Mnemonic:
        return self._index[s.strip().lower()]

# %%
if __name__ == '__main__':
   
    M = Mnemonics()
    M["kidney"]
