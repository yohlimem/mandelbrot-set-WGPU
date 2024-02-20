struct Camera{
    offset: vec2<f32>,
    zoom: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}



@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}


// Fragment shader
// group(0): texture_bind_group_layout
//  binding(0): bindGroupLayoutEntry binding 0
@group(0) @binding(0) 
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@group(1) @binding(0)
var<uniform> camera: Camera;
// https://sotrh.github.io/learn-wgpu/beginner/tutorial5-textures/#the-results
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_coords: vec2<f32> = in.tex_coords;
    
    var z = vec2<f32>(0.0,0.0);
    var c = (tex_coords - 0.5) * camera.zoom - camera.offset;

    for (var i = 0; i < 500; i++){
        z = mult_complex(z,z) + c;

        if (length(z) > 2.0){
            return vec4<f32>(f32(i)/500.0, f32(i)/500.0, f32(i)/500.0 , 1.0);
        }
    }


    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

fn mult_complex(complex: vec2<f32>, other: vec2<f32>) -> vec2<f32> {


    // let angle_other = atan(other.y/other.x);

    // let angle = atan(complex.y/complex.x);

    // let total_angle = angle + angle_other;

    // let magnitude = length(complex) * length(other);

    // return vec2<f32>(magnitude * cos(total_angle), magnitude * sin(total_angle));

    return vec2<f32>(complex.x * other.x - complex.y * other.y, complex.x * other.y + complex.y * other.x);


}
