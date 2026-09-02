# Prior art and related work

This is a research-oriented map, **not a novelty opinion or patent search**.
Symbolic-U does not claim that its individual ingredients are unprecedented.
The interesting question is whether the project's particular unification under a
small Key/U substrate is useful and experimentally distinguishable from its
closest predecessors.

## Strong related lines

| Research line | Relevant overlap with Symbolic-U | Important difference / caution |
|---|---|---|
| STRIPS / Shakey | symbolic world models, goal-directed planning, robot perception-to-reasoning pipeline | does not by itself establish the project's learning method |
| Ulysses selective perception | reasoning chooses which perceptual information matters for action; persistent world model | much of the task/domain model is explicitly engineered |
| RoboSherlock | task-relevant perception queries, multiple algorithms, knowledge-driven pipelines | perception capabilities and knowledge infrastructure are more pre-engineered |
| KnowRob | symbolic knowledge processing connected to robot perception/actions and on-demand sources | broader knowledge infrastructure, different learning/execution semantics |
| Soar chunking | successful reasoning is backtraced and compiled into reusable rules | base productions and architecture differ; Macro-U revision semantics are not identical |
| TMS / ATMS | reasons, dependencies, belief revision, inconsistency/context handling | Symbolic-U separates proposition state from derivation state and combines this with learned/compiled U |
| Belnap-Dunn style four-valued evidence | distinguishes positive, negative, neither and both information | Symbolic-U encodes proposition status with ternary Key state plus contradiction flag; this is not a new logic claim |
| ILP / Meta-Interpretive Learning | learns symbolic rules/programs, recursion and predicate invention from examples/background knowledge | Symbolic-U's exact curriculum, U state semantics and cross-domain runtime are different design choices |
| Perceptual anchoring | connects persistent symbols to sensor observations of physical objects | Symbolic-U's current visual experiments are simpler synthetic scenes |
| DyKnow | bridges noisy/quantitative sensing and crisp symbolic reasoning across abstraction layers | different middleware and semantics |
| Metareasoning | selects computations under resource limits | Symbolic-U currently uses simpler symbolic cost/control mechanisms |
| Apperception Engine | induces explicit interpretable relational theories from sensory sequences | later raw-input work uses a neural frontend; Symbolic-U currently uses classical sensors |

## Selected references

1. R. E. Fikes and N. J. Nilsson, **STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving**, *Artificial Intelligence* 2(3-4), 1971. DOI: [10.1016/0004-3702(71)90010-5](https://doi.org/10.1016/0004-3702(71)90010-5).
2. D. A. Reece, **Selective Perception for Robot Driving**, PhD thesis, Carnegie Mellon University, 1992, CMU-CS-92-139. CMU's thesis summary describes Ulysses-2 choosing data to sense from an inference tree and Ulysses-3 adding a persistent time-stamped world model.
3. M. Beetz et al., **RoboSherlock: Unstructured Information Processing for Robot Perception**, ICRA 2015. DOI: [10.1109/ICRA.2015.7139395](https://doi.org/10.1109/ICRA.2015.7139395).
4. M. Tenorth and M. Beetz, **KnowRob: A Knowledge Processing Infrastructure for Cognition-enabled Robots**, *IJRR* 32(5), 2013. DOI: [10.1177/0278364913481635](https://doi.org/10.1177/0278364913481635).
5. J. E. Laird, P. S. Rosenbloom and A. Newell, **Chunking in Soar: The Anatomy of a General Learning Mechanism**, *Machine Learning* 1, 1986. DOI: [10.1007/BF00116249](https://doi.org/10.1007/BF00116249).
6. J. Doyle, **A Truth Maintenance System**, *Artificial Intelligence* 12(3), 1979. DOI: [10.1016/0004-3702(79)90008-0](https://doi.org/10.1016/0004-3702(79)90008-0).
7. J. de Kleer, **An Assumption-based TMS**, *Artificial Intelligence* 28(2), 1986. DOI: [10.1016/0004-3702(86)90080-9](https://doi.org/10.1016/0004-3702(86)90080-9).
8. N. D. Belnap, **A Useful Four-Valued Logic**, in *Modern Uses of Multiple-Valued Logic*, 1977. DOI: [10.1007/978-94-010-1161-7_2](https://doi.org/10.1007/978-94-010-1161-7_2).
9. S. H. Muggleton, **Inductive Logic Programming**, *New Generation Computing* 8(4), 1991. DOI: [10.1007/BF03037089](https://doi.org/10.1007/BF03037089).
10. S. H. Muggleton, D. Lin and A. Tamaddoni-Nezhad, **Meta-Interpretive Learning of Higher-Order Dyadic Datalog: Predicate Invention Revisited**, *Machine Learning* 100, 2015. DOI: [10.1007/s10994-014-5471-y](https://doi.org/10.1007/s10994-014-5471-y).
11. S. Coradeschi and A. Saffiotti, **An Introduction to the Anchoring Problem**, *Robotics and Autonomous Systems* 43, 2003. DOI: [10.1016/S0921-8890(03)00021-6](https://doi.org/10.1016/S0921-8890(03)00021-6).
12. F. Heintz, J. Kvarnström and P. Doherty, **Bridging the Sense-Reasoning Gap: DyKnow**, *Advanced Engineering Informatics* 24(1), 2010. DOI: [10.1016/j.aei.2009.08.007](https://doi.org/10.1016/j.aei.2009.08.007).
13. S. Russell and E. Wefald, **Principles of Metareasoning**, *Artificial Intelligence* 49, 1991. DOI: [10.1016/0004-3702(91)90015-C](https://doi.org/10.1016/0004-3702(91)90015-C).
14. R. Evans et al., **Making Sense of Raw Input**, *Artificial Intelligence* 299, 2021. DOI: [10.1016/j.artint.2021.103521](https://doi.org/10.1016/j.artint.2021.103521).

## What should not be claimed

The repository should not claim to be the first system to:

- combine sensors with symbolic reasoning;
- perform query- or task-driven perception;
- learn symbolic rules;
- learn recursive logic programs;
- maintain justification/dependency information;
- compile successful reasoning into reusable rules;
- represent inconsistent and incomplete information;
- use a classical perception frontend above a symbolic learner.

## Current differentiating research question

A more defensible question is:

> Can a small, explicitly ablated Key/U substrate support the same learned and
> revisable mechanism across language grounding, recursive arithmetic, temporal
> state, visual semantics, sensor selection and proofgraph compilation while
> retaining explicit unknown/contradiction/provenance behavior?

That is an empirical unification question, not a historical priority claim.
