`timescale 1ns / 1ps

module pointwise_conv_unit #(
    parameter DATA_W    = 8,
    parameter ACC_W     = 32,
    parameter IN_CH     = 4,
    parameter OUT_CH    = 8,
    parameter ACC_REG_W = 48
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      i_valid,
    input  wire signed [IN_CH*ACC_W-1:0]           i_vec_flat,
    input  wire signed [OUT_CH*IN_CH*DATA_W-1:0] i_weights_flat,
    
    output reg  signed [OUT_CH*ACC_W-1:0]   o_vec_flat,
    output reg                           o_valid
);

    // -- FSM ״̬���� --
    localparam S_IDLE   = 2'b00; // ����״̬���ȴ�����
    localparam S_CALC   = 2'b01; // ����״̬��ִ�г˼�
    localparam S_OUTPUT = 2'b10; // ���״̬���ṩ���

    // -- �ڲ�������Ĵ��� --
    wire signed [DATA_W-1:0] i_weights [0:OUT_CH*IN_CH-1];
    genvar gi;
    generate
        for (gi = 0; gi < OUT_CH*IN_CH; gi = gi + 1) begin : UNPACK_WEIGHTS
            assign i_weights[gi] = i_weights_flat[(gi+1)*DATA_W-1 -: DATA_W];
        end
    endgenerate

    reg signed [ACC_W-1:0]     i_vec_reg [0:IN_CH-1];
    reg signed [ACC_REG_W-1:0] acc_regs [0:OUT_CH-1];
    reg [1:0]                  state; // FSM ״̬�Ĵ���
    reg [$clog2(IN_CH)-1:0]    icnt;  // ����ͨ��������
    integer                    oc;

    // ---- ����״̬��������·�� ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ��λ���мĴ�����״̬
            state      <= S_IDLE;
            icnt       <= 0;
            o_valid    <= 1'b0;
            o_vec_flat <= 0;
            for (oc = 0; oc < IN_CH; oc = oc + 1)
                i_vec_reg[oc] <= 0;
            for (oc = 0; oc < OUT_CH; oc = oc + 1)
                acc_regs[oc] <= 0;
        end else begin
            // FSM ״̬ת���߼�
            case (state)
                S_IDLE: begin
                    o_valid <= 1'b0; // Ĭ����Ч���
                    if (i_valid) begin
                        // ���յ���Ч���룬������������
                        for (oc = 0; oc < IN_CH; oc = oc + 1) begin
                            i_vec_reg[oc] <= i_vec_flat[(oc+1)*ACC_W-1 -: ACC_W];
                        end
                        icnt  <= 0;       // ������������
                        state <= S_CALC;  // ��ת������״̬
                    end
                end
                S_CALC: begin
                    o_valid <= 1'b0; // ��������У������Ч
                    if (icnt == 0) begin
                        for (oc = 0; oc < OUT_CH; oc = oc + 1) begin
                            acc_regs[oc] <= $signed(i_vec_reg[0]) * $signed({{(ACC_REG_W-DATA_W){i_weights[oc*IN_CH + 0][DATA_W-1]}}, i_weights[oc*IN_CH + 0]});
                        end
                    end else begin
                        // �������ڣ�ִ���ۼ�
                        for (oc = 0; oc < OUT_CH; oc = oc + 1) begin
                            acc_regs[oc] <= acc_regs[oc] + 
                                         ($signed(i_vec_reg[icnt]) * $signed({{(ACC_REG_W-DATA_W){i_weights[oc*IN_CH + icnt][DATA_W-1]}}, i_weights[oc*IN_CH + icnt]}));
                        end
                    end

                    // �жϼ����Ƿ����
                    if (icnt == IN_CH - 1) begin
                        state <= S_OUTPUT; // ������ɣ���ת�����״̬
                    end else begin
                        icnt <= icnt + 1; // �������㣬��������һ
                    end
                end

                S_OUTPUT: begin
                    o_valid    <= 1'b1; // �������Ч�ź��ø�
                    // �����ս����������˿�
                    for (oc = 0; oc < OUT_CH; oc = oc + 1) begin
                        o_vec_flat[(oc+1)*ACC_W-1 -: ACC_W] <= acc_regs[oc][ACC_W-1:0];
                    end
                    state <= S_IDLE; // �����ɣ����ؿ���״̬
                end
                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
