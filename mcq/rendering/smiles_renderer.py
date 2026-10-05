"""SMILES chemical-structure renderer using SmilesDrawer (client-side JS, 0ms)."""

import html
import streamlit.components.v1 as components

_SMILES_JS = "https://cdn.jsdelivr.net/npm/smiles-drawer@2.1.7/dist/smiles-drawer.min.js"


def render_chemical_structure(smiles: str, name: str | None = None, height: int = 220) -> None:
    """Render a SMILES string as a 2-D structure using SmilesDrawer (canvas)."""
    safe_smiles = html.escape(smiles, quote=True)
    safe_name   = html.escape(name or "", quote=True)

    html_code = f"""<!DOCTYPE html><html>
<head><meta charset="UTF-8">
<script src="{_SMILES_JS}"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:transparent;display:flex;flex-direction:column;align-items:center;justify-content:center;height:{height}px}}
  canvas{{max-width:100%}}
  .mol-name{{font-family:'EB Garamond',Georgia,serif;font-size:.9rem;color:#4a4440;margin-top:6px;font-weight:500}}
  .mol-err{{font-family:monospace;font-size:.8rem;color:#8b2020;padding:8px}}
</style>
</head>
<body>
<canvas id="molCanvas"></canvas>
{"<div class='mol-name'>" + safe_name + "</div>" if name else ""}
<script>
window.addEventListener("DOMContentLoaded", function() {{
  try {{
    var options = {{
      width: 320,
      height: {height - 35},
      bondThickness: 1.5,
      bondColor: "#2c2825",
      terminalCarbons: true
    }};
    var drawer = new SmilesDrawer.Drawer(options);
    SmilesDrawer.parse("{safe_smiles}", function(tree) {{
      drawer.draw(tree, "molCanvas", "light", false);
    }}, function(err) {{
      console.error(err);
    }});
  }} catch(e) {{
    console.error(e);
  }}
}});
</script>
</body></html>"""

    components.html(html_code, height=height, scrolling=False)
