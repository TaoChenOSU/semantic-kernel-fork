// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json.Serialization;
using SemanticKernel.Service.Storage;

namespace SemanticKernel.Service.Model;

/// <summary>
/// A chat invitation for a user to join a chat session.
/// This will also be used to manage participants in a chat session.
/// </summary>
public class ChatInvitation : IStorageEntity
{
    /// <summary>
    /// The ID of the invitation.
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; set; }

    /// <summary>
    /// The user ID of the user who is invited to the chat session.
    /// </summary>
    [JsonPropertyName("userId")]
    public string UserId { get; set; }

    /// <summary>
    /// The chat ID of the chat session the user is invited to.
    /// </summary>
    [JsonPropertyName("chatId")]
    public string ChatId { get; set; }

    /// <summary>
    /// Whether the invitation has been accepted.
    /// </summary>
    [JsonPropertyName("isAccepted")]
    public bool IsAccepted { get; set; }

    public ChatInvitation(string userId, string chatId)
    {
        this.Id = Guid.NewGuid().ToString();
        this.UserId = userId;
        this.ChatId = chatId;
        this.IsAccepted = false;
    }
}