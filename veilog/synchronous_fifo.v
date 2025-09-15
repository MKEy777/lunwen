`timescale 1ns / 1ps

module synchronous_fifo #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 16
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // Write Interface
    input  wire                      i_wr_en,
    input  wire [DATA_WIDTH-1:0]     i_wdata,
    output wire                      o_full,

    // Read Interface
    input  wire                      i_rd_en,
    output wire [DATA_WIDTH-1:0]     o_rdata,
    output wire                      o_empty
);
    localparam ADDR_WIDTH = $clog2(DEPTH);

    // �洢������
    (* ram_style = "block" *) // ָ���ۺ���ʹ��BRAM
    reg [DATA_WIDTH-1:0] mem[0:DEPTH-1];
    
    // ��дָ�룬�ȵ�ַλ���1λ������������/��״̬
    reg [ADDR_WIDTH:0]   wr_ptr;
    reg [ADDR_WIDTH:0]   rd_ptr;

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = 0;
        end
    end

    // --- �����߼� ---
    // ָ��Ԥ����
    wire [ADDR_WIDTH:0] wr_ptr_next = wr_ptr + 1;
    wire [ADDR_WIDTH:0] rd_ptr_next = rd_ptr + 1;

    // ��/��״̬�ж� (�߼����ֲ���)
    assign o_full  = (wr_ptr_next == {~rd_ptr[ADDR_WIDTH], rd_ptr[ADDR_WIDTH-1:0]});
    assign o_empty = (wr_ptr == rd_ptr);
    
    // �����ݶ˿� (��Ϊ���ֲ���)
    assign o_rdata = mem[rd_ptr[ADDR_WIDTH-1:0]];

    // --- д�߼� (�޸�Ϊͬ����λ) ---
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (i_wr_en && !o_full) begin
            mem[wr_ptr[ADDR_WIDTH-1:0]] <= i_wdata;
            wr_ptr <= wr_ptr_next;
        end
    end

    // --- ���߼� (�޸�Ϊͬ����λ) ---
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_ptr <= 0;
        end else if (i_rd_en && !o_empty) begin
            rd_ptr <= rd_ptr_next;
        end
    end
    
endmodule