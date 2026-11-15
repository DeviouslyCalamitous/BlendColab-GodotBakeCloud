@tool
extends EditorPlugin

## Automates LightmapGI baking inside a Vulkan-enabled Xvfb Linux container.
## Remains completely dormant during normal interactive editing unless
## `--bake-scene=res://...` is passed after `--` on the Godot command line.

var _target_scene: String = ""
var _report_path: String = ""
var _is_running: bool = false


func _enter_tree() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--bake-scene="):
			_target_scene = arg.trim_prefix("--bake-scene=")
		elif arg.begins_with("--bake-report="):
			_report_path = arg.trim_prefix("--bake-report=")

	if _target_scene.is_empty():
		return  # Normal editor session; do nothing

	call_deferred("_run_automated_bake")


func _run_automated_bake() -> void:
	if _is_running:
		return
	_is_running = true

	var start_msec := Time.get_ticks_msec()
	print("[GodotBakeCloud] Initializing cloud bake for: ", _target_scene)

	# Verify Vulkan RenderingDevice is active (fails if launched with --headless instead of Xvfb)
	var rd := RenderingServer.get_rendering_device()
	if rd == null:
		_fail_and_quit("RenderingDevice is null. Launch Godot with --rendering-driver vulkan under xvfb-run, not --headless.")
		return

	print("[GodotBakeCloud] Active GPU: ", rd.get_device_name())

	if not ResourceLoader.exists(_target_scene):
		_fail_and_quit("Target scene does not exist: " + _target_scene)
		return

	EditorInterface.open_scene_from_path(_target_scene)
	await get_tree().process_frame
	await get_tree().process_frame

	var root := EditorInterface.get_edited_scene_root()
	if root == null:
		_fail_and_quit("Failed to open scene root for: " + _target_scene)
		return

	var lightmap_node := _find_lightmap_gi(root)
	if lightmap_node == null:
		_fail_and_quit("No LightmapGI node found in scene: " + _target_scene)
		return

	# Selecting the LightmapGI node activates Godot's built-in LightmapGIEditorPlugin toolbar
	EditorInterface.edit_node(lightmap_node)
	await get_tree().process_frame
	await get_tree().process_frame

	var bake_button := _find_bake_button(EditorInterface.get_base_control())
	if bake_button == null:
		_fail_and_quit("Could not locate 'Bake Lightmaps' toolbar button in EditorInterface.")
		return

	# Ensure output path is set on LightmapGI so .lmbake saves cleanly into project/Drive
	var expected_lmbake := _target_scene.get_basename() + ".lmbake"
	if lightmap_node.light_data != null and not lightmap_node.light_data.resource_path.is_empty():
		expected_lmbake = lightmap_node.light_data.resource_path

	var prev_mtime := FileAccess.get_modified_time(expected_lmbake) if FileAccess.file_exists(expected_lmbake) else 0

	print("[GodotBakeCloud] Triggering LightmapGI bake...")
	bake_button.pressed.emit()

	# Poll until .lmbake file is written/updated (with 30-minute safety ceiling)
	var timeout_msec := 30 * 60 * 1000
	var baked := false
	while (Time.get_ticks_msec() - start_msec) < timeout_msec:
		await get_tree().create_timer(1.0).timeout
		if FileAccess.file_exists(expected_lmbake):
			var new_mtime := FileAccess.get_modified_time(expected_lmbake)
			if new_mtime > prev_mtime:
				baked = true
				break

	if not baked:
		_fail_and_quit("Timed out waiting for .lmbake output at: " + expected_lmbake)
		return

	# Allow editor file system scan to settle and save the updated scene reference
	await get_tree().create_timer(1.5).timeout
	EditorInterface.save_scene()
	await get_tree().process_frame

	var elapsed_sec := (Time.get_ticks_msec() - start_msec) / 1000.0
	print("[GodotBakeCloud] Bake finished in %.2fs -> %s" % [elapsed_sec, expected_lmbake])

	_write_report({
		"status": "ok",
		"scene": _target_scene,
		"lightmap_file": expected_lmbake,
		"gpu_device": rd.get_device_name(),
		"bake_duration_sec": snappedf(elapsed_sec, 0.01),
	})
	get_tree().quit(0)


func _find_lightmap_gi(node: Node) -> LightmapGI:
	if node is LightmapGI:
		return node
	for child in node.get_children():
		var found := _find_lightmap_gi(child)
		if found != null:
			return found
	return null


func _find_bake_button(root: Node) -> Button:
	if root is Button:
		var txt := (root as Button).text.strip_edges()
		if txt == "Bake Lightmaps" or txt == tr("Bake Lightmaps"):
			return root as Button
	for child in root.get_children():
		var found := _find_bake_button(child)
		if found != null:
			return found
	return null


func _write_report(data: Dictionary) -> void:
	if _report_path.is_empty():
		return
	var f := FileAccess.open(_report_path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(data, "\t"))
		f.close()


func _fail_and_quit(reason: String) -> void:
	push_error("[GodotBakeCloud] " + reason)
	_write_report({"status": "error", "reason": reason})
	get_tree().quit(1)