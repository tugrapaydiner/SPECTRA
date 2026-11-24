"""Checked two-plan migration support.

Given a conflict graph and two proper colourings ``A`` and ``B``, every vertex may
keep ``A(v)`` or move to ``B(v)``, and each query restricts arbitrary original
vertices. A checked service returns either a complete assignment that contains the
unique inclusion-minimal migration closure, or an original-edge implication path
proving the request contradictory. Solver and observer build independent native
indexes; native code is compiled only by an explicit ``build`` call.

``PacketService`` validates immutable request packets with two independent native
parsers, ``CompactService`` uses the sparse SCC engine, and ``StreamedService``
pairs it with a streamed observer.
"""
from .compact_service import CompactService
from .packet_service import PacketService
from .streamed_service import StreamedService

__all__ = ["CompactService", "PacketService", "StreamedService"]
