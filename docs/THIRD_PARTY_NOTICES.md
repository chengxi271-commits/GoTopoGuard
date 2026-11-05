# Third-party notices

Original GoTopoGuard Python code is Apache-2.0. The separately distributed
patches contain modifications and context from the following upstream projects
and retain the applicable upstream terms:

| Material | Upstream | License text |
|---|---|---|
| `leela_zero_gotopo_set_position.patch` | AlignmentResearch/leela-zero | [GPLv3](../licenses/LeelaZero-COPYING.txt) |
| `elf_opengo_*.patch` | AlignmentResearch/ELF | [BSD](../licenses/ELF-LICENSE.txt) |
| `katago_custom_opencl_gcc11.patch` | AlignmentResearch/KataGo-custom | [MIT](../licenses/KataGo-MIT.txt) |
| Go Attack used by reproduction commands | AlignmentResearch/go_attack | [MIT notice](../licenses/GoAttack-MIT.txt) |

Exact commits and patch order are documented in [ENGINES.md](ENGINES.md).
No upstream engine source tree, binary, model weight, CUDA dependency, or SGF
collection is bundled. Calling an external engine does not relicense it under
GoTopoGuard's Apache license. Python analysis dependencies likewise retain
their own licenses and are installed rather than vendored.

Springer paper files, templates, and signed publication agreements are outside
this source release and are not covered by its software license.
