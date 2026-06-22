import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DNA Sequence Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 DNA Sequence Analyzer")
st.markdown("Analyze DNA sequences — GC content, "
            "codons, mutations and pattern matching.")
st.markdown("---")

# Codon table
CODON_TABLE = {
    'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
    'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met (Start)',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'STOP', 'TAG': 'STOP',
    'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys', 'TGA': 'STOP', 'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
    'GTA': 'Val', 'TGG': 'Trp'
}

SAMPLE_SEQUENCES = {
    "Sample Human Gene (TP53)":
        "ATGTTCAAGACAGATTTTCAACTCTGTCTCCTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTC",
    "Sample Bacterial Gene":
        "ATGAAAAAAATCGCCGTTATCGGAGCAGTCGCCGTTGGAGGCGTCATCGCTGCTTTTGAAATTCAGAAAGCAAAAGATGCTGAAGGTAAACTGATCGAAGACGGTAAAGTCATCGAAGTTGAAGACTTCAAAGTTGTTGAAGACGAAATCGAAATCAAAGACGAAGGT",
    "Random Sequence":
        "ATGGCTAGCTAGCGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTAGCGCGATCGATCGATCGATCGATCGATCGATCGATCG"
}

def clean_sequence(seq):
    return re.sub(r'[^ATCGN]', '',
                  seq.upper().replace(' ', '')
                  .replace('\n', ''))

def gc_content(seq):
    if not seq:
        return 0
    gc = seq.count('G') + seq.count('C')
    return round(gc / len(seq) * 100, 2)

def complement(seq):
    comp = {'A': 'T', 'T': 'A',
            'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in seq)

def reverse_complement(seq):
    return complement(seq)[::-1]

def find_codons(seq):
    codons = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            codons.append({
                'position': i + 1,
                'codon': codon,
                'amino_acid': CODON_TABLE.get(
                    codon, 'Unknown')
            })
    return codons

def find_orfs(seq):
    orfs = []
    for i in range(len(seq) - 2):
        if seq[i:i+3] == 'ATG':
            for j in range(i+3, len(seq)-2, 3):
                codon = seq[j:j+3]
                if codon in ['TAA', 'TAG', 'TGA']:
                    if j - i >= 60:
                        orfs.append({
                            'start': i + 1,
                            'end': j + 3,
                            'length': j + 3 - i,
                            'sequence': seq[i:j+3]
                        })
                    break
    return orfs

def find_pattern(seq, pattern):
    pattern = pattern.upper()
    matches = []
    start = 0
    while True:
        idx = seq.find(pattern, start)
        if idx == -1:
            break
        matches.append(idx + 1)
        start = idx + 1
    return matches

def gc_window(seq, window=50):
    if len(seq) < window:
        return [], []
    positions, gc_vals = [], []
    for i in range(0, len(seq) - window, window//2):
        window_seq = seq[i:i+window]
        positions.append(i + window//2)
        gc_vals.append(gc_content(window_seq))
    return positions, gc_vals

# Sidebar
st.sidebar.header("🧬 Input Sequence")

input_method = st.sidebar.radio(
    "Input method:",
    ["Sample Sequences", "Enter Custom DNA"])

if input_method == "Sample Sequences":
    sample_name = st.sidebar.selectbox(
        "Choose sample:",
        list(SAMPLE_SEQUENCES.keys()))
    raw_seq = SAMPLE_SEQUENCES[sample_name]
else:
    raw_seq = st.sidebar.text_area(
        "Paste DNA sequence (A, T, C, G only):",
        height=200,
        placeholder="ATCGATCGATCG..."
    )

sequence = clean_sequence(raw_seq) if raw_seq else ""

if not sequence:
    st.info("👈 Select a sample or paste a DNA "
            "sequence in the sidebar to begin.")
    st.stop()

# Basic stats
st.markdown(
    f"### Analyzing: `{sequence[:50]}"
    f"{'...' if len(sequence)>50 else ''}`")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Length", f"{len(sequence)} bp")
c2.metric("GC Content", f"{gc_content(sequence)}%")
c3.metric("A Count", sequence.count('A'))
c4.metric("T Count", sequence.count('T'))
c5.metric("G+C Count",
          sequence.count('G') + sequence.count('C'))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Composition",
    "🔬 Codons",
    "🧪 ORF Finder",
    "🔍 Pattern Search",
    "📈 GC Window"
])

# Tab 1 — Composition
with tab1:
    st.markdown("### 📊 Nucleotide Composition")

    col1, col2 = st.columns(2)

    with col1:
        counts = {
            'A': sequence.count('A'),
            'T': sequence.count('T'),
            'G': sequence.count('G'),
            'C': sequence.count('C')
        }
        fig = px.bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            title='Nucleotide Frequency',
            color=list(counts.keys()),
            color_discrete_map={
                'A': '#e74c3c', 'T': '#3498db',
                'G': '#2ecc71', 'C': '#f39c12'
            }
        )
        fig.update_layout(
            height=350, template='plotly_white',
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            values=list(counts.values()),
            names=list(counts.keys()),
            title='Nucleotide Distribution',
            color_discrete_map={
                'A': '#e74c3c', 'T': '#3498db',
                'G': '#2ecc71', 'C': '#f39c12'
            }
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🔄 Complementary Strands")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**5' → 3' Original:**")
        st.code(sequence[:80] +
                ('...' if len(sequence) > 80 else ''))
    with col2:
        st.markdown("**3' → 5' Complement:**")
        st.code(complement(sequence)[:80] +
                ('...' if len(sequence) > 80 else ''))
    st.markdown("**Reverse Complement:**")
    st.code(reverse_complement(sequence)[:80] +
            ('...' if len(sequence) > 80 else ''))

    gc = gc_content(sequence)
    if gc < 35:
        st.info("⚠️ Low GC content — AT-rich region, "
                "common in regulatory regions.")
    elif gc > 65:
        st.warning("⚠️ High GC content — may indicate "
                   "CpG island or coding region.")
    else:
        st.success(f"✅ GC content {gc}% is in the "
                   f"typical range (35–65%).")

# Tab 2 — Codons
with tab2:
    st.markdown("### 🔬 Codon Analysis")

    codons = find_codons(sequence)
    if codons:
        codon_df = pd.DataFrame(codons)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{len(codons)} codons found**")
            st.dataframe(
                codon_df.head(20),
                use_container_width=True,
                hide_index=True)

        with col2:
            aa_counts = codon_df[
                'amino_acid'].value_counts().head(10)
            fig3 = px.bar(
                x=aa_counts.values,
                y=aa_counts.index,
                orientation='h',
                title='Top 10 Amino Acids Encoded',
                color=aa_counts.values,
                color_continuous_scale='Viridis'
            )
            fig3.update_layout(
                height=350, template='plotly_white')
            st.plotly_chart(
                fig3, use_container_width=True)

        stop_codons = codon_df[
            codon_df['amino_acid'] == 'STOP']
        start_codons = codon_df[
            codon_df['codon'] == 'ATG']

        c1, c2, c3 = st.columns(3)
        c1.metric("Start Codons (ATG)",
                  len(start_codons))
        c2.metric("Stop Codons",
                  len(stop_codons))
        c3.metric("Unique Codons",
                  codon_df['codon'].nunique())

# Tab 3 — ORF Finder
with tab3:
    st.markdown("### 🧪 Open Reading Frame (ORF) Finder")
    st.markdown("ORFs are sequences between a start "
                "codon (ATG) and stop codon.")

    orfs = find_orfs(sequence)

    if orfs:
        st.success(f"✅ Found {len(orfs)} ORFs "
                   f"(minimum 60 bp)")
        orf_df = pd.DataFrame(orfs)[[
            'start', 'end', 'length']]
        orf_df.columns = [
            'Start (bp)', 'End (bp)', 'Length (bp)']

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(orf_df,
                         use_container_width=True,
                         hide_index=True)
        with col2:
            fig4 = px.bar(
                orf_df,
                x='Start (bp)', y='Length (bp)',
                title='ORF Lengths by Position',
                color='Length (bp)',
                color_continuous_scale='Greens'
            )
            fig4.update_layout(
                height=350, template='plotly_white')
            st.plotly_chart(
                fig4, use_container_width=True)

        st.markdown("#### Longest ORF")
        longest = max(orfs, key=lambda x: x['length'])
        st.markdown(
            f"**Position:** {longest['start']} — "
            f"{longest['end']} "
            f"({longest['length']} bp)")
        st.code(longest['sequence'][:100] +
                ('...' if len(longest['sequence'])
                 > 100 else ''))
    else:
        st.info("No significant ORFs found "
                "(minimum 60 bp threshold).")

# Tab 4 — Pattern Search
with tab4:
    st.markdown("### 🔍 Pattern / Motif Search")

    pattern_input = st.text_input(
        "Enter DNA pattern to search:",
        placeholder="e.g. ATCG or TATAAA")

    col1, col2 = st.columns(2)
    with col1:
        common_patterns = {
            "TATA Box (promoter)": "TATAAA",
            "Kozak Sequence": "ACCATG",
            "CpG Site": "CG",
            "EcoRI site": "GAATTC",
            "BamHI site": "GGATCC"
        }
        selected_common = st.selectbox(
            "Or pick a known motif:",
            ["Custom"] + list(common_patterns.keys()))
        if selected_common != "Custom":
            pattern_input = common_patterns[
                selected_common]

    if pattern_input:
        pattern = clean_sequence(pattern_input)
        if pattern:
            positions = find_pattern(sequence, pattern)
            with col2:
                st.metric("Occurrences Found",
                          len(positions))

            if positions:
                st.success(
                    f"✅ Pattern `{pattern}` found "
                    f"at positions: "
                    f"{', '.join(map(str, positions[:20]))}"
                    f"{'...' if len(positions) > 20 else ''}")

                # Visualize
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=positions,
                    y=[1]*len(positions),
                    mode='markers',
                    marker=dict(
                        symbol='line-ns',
                        size=20,
                        color='#e74c3c',
                        line=dict(
                            width=2,
                            color='#e74c3c')
                    ),
                    name=f'Pattern: {pattern}'
                ))
                fig5.update_layout(
                    title=f'Pattern `{pattern}` '
                          f'Locations in Sequence',
                    xaxis_title='Position (bp)',
                    height=200,
                    template='plotly_white',
                    yaxis_visible=False
                )
                st.plotly_chart(
                    fig5, use_container_width=True)
            else:
                st.warning(
                    f"Pattern `{pattern}` not "
                    f"found in sequence.")

# Tab 5 — GC Window
with tab5:
    st.markdown("### 📈 GC Content Along Sequence")
    st.markdown("Sliding window GC analysis reveals "
                "variation across the sequence.")

    window_size = st.slider(
        "Window size (bp):", 20, 200, 50, 10)

    positions, gc_vals = gc_window(
        sequence, window_size)

    if positions:
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=positions, y=gc_vals,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#2ecc71', width=2),
            fillcolor='rgba(46,204,113,0.2)'
        ))
        fig6.add_hline(
            y=50, line_dash="dash",
            line_color="gray",
            annotation_text="50% GC")
        fig6.update_layout(
            title=f'GC Content — '
                  f'{window_size}bp Window',
            xaxis_title='Position (bp)',
            yaxis_title='GC Content (%)',
            yaxis_range=[0, 100],
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig6,
                        use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Min GC", f"{min(gc_vals):.1f}%")
        col2.metric("Max GC", f"{max(gc_vals):.1f}%")
        col3.metric("Avg GC", f"{np.mean(gc_vals):.1f}%")
    else:
        st.info("Sequence too short for "
                "window analysis.")

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "DNA Sequence Analyzer | "
    "Bioinformatics + Python"
)