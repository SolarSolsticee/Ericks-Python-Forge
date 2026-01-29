# ... (Your existing layout code for col_controls and col_plot) ...

    with col_controls:
        # ... (Strain Min/Max inputs) ...
        # ... (IV input) ...
        # ... (Tag selection) ...
        
        notes = st.text_area("Notes", placeholder="e.g. 50% infill", key='input_notes')
        
        # --- SAVE CONFIRMATION LOGIC ---
        if "confirm_save" not in st.session_state:
            st.session_state["confirm_save"] = False

        # --- THIS IS WHERE THE SAVE BUTTON BLOCK GOES ---
        submit = st.button("Calculate & Save to DB", type="primary")

        if submit:
            exists = check_sheet_exists(Path(output_csv), sanitized_name)
            if exists:
                st.session_state["confirm_save"] = True
            else:
                st.session_state["do_save"] = True

        if st.session_state["confirm_save"]:
            st.warning(f"⚠️ Data for **{sanitized_name}** already exists.")
            col_c1, col_c2 = st.columns(2)
            if col_c1.button("✅ Overwrite"):
                st.session_state["do_save"] = True
                st.session_state["confirm_save"] = False
            if col_c2.button("❌ Cancel"):
                st.session_state["confirm_save"] = False
                st.info("Cancelled.")

    with col_plot:
        # ... (Your existing plotting code) ...
        fig, ax = plt.subplots(figsize=(8, 5))
        # ... sns.lineplot ...
        st.pyplot(fig)

    # ==========================================
    # === SAVE EXECUTION BLOCK (Indent Level 1) ===
    # ==========================================
    # This logic sits outside the columns but inside the "if selected_sheet:" block
    
    if st.session_state.get("do_save", False):
        rows = []
        summary_metrics = []
        
        # COMBINE TAGS (Dropdown + Text)
        final_tag_list = selected_tags.copy()
        if new_tags_input:
            manual_tags = [t.strip() for t in new_tags_input.split(',') if t.strip()]
            final_tag_list.extend(manual_tags)
        # Deduplicate and join
        final_tag_string = ", ".join(sorted(list(set(final_tag_list))))
        
        for p in plot_data:
            strain = p['strain_raw']
            stress = p['stress_raw']
            
            # ... (Modulus Fit Logic) ...
            limit_min = strain_min_pct / 100.0
            limit_max = strain_max_pct / 100.0
            
            mask = (strain >= limit_min) & (strain <= limit_max)
            fit = fit_modulus(strain[mask], stress[mask])
            modulus_pa = fit['slope']
            modulus_gpa = modulus_pa / 1e9 if not np.isnan(modulus_pa) else 0.0
            
            # ... (Curve Serialization Logic) ...
            # ... (Copy-paste the downsampling/cleaning code I gave you earlier here) ...

            rows.append({
                'timestamp': datetime.now().isoformat(), # <--- ENSURE THIS IS HERE
                'workbook_filename': filename,
                'sheet_name_raw': selected_sheet,
                'sheet_name_sanitized': sanitized_name,
                'sample_name': f'{sanitized_name}-{p["id"]}',
                'sample_index': p['id'],
                'initial_length_mm': initial_length_mm,
                'thickness_mm': thickness_mm,
                'width_mm': width_mm,
                'area_mm2': width_mm * thickness_mm,
                'modulus_pa': modulus_pa,
                'modulus_gpa': modulus_gpa,
                'r_value': fit['rvalue'],
                'n_points': fit['n'],
                'strain_fit_min': limit_min,
                'strain_fit_max': limit_max,
                'notes': notes,
                'tags': final_tag_string, 
                'iv': iv_val if iv_val > 0 else None,
                'curve_strain': curve_strain_str,
                'curve_stress': curve_stress_str,
            })
            summary_metrics.append(modulus_gpa)

        update_database(Path(output_csv), rows)
        st.session_state["do_save"] = False
        
        avg_mod = np.nanmean(summary_metrics)
        st.success(f"Saved {len(rows)} samples to `{output_csv}`")
        st.metric("Average Modulus", f"{avg_mod:.2f} GPa")
        st.dataframe(pd.DataFrame(rows)[['sample_name', 'modulus_gpa', 'tags']])
